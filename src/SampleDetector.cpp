#include <sys/stat.h>
#include <fstream>
#include <glog/logging.h>
#include <dlfcn.h>

#include "SampleDetector.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "ji_utils.h"
#include "./logging.h"
#include "NvInferPlugin.h"

#define INPUT_NAME "images"
#define OUTPUT_NAME "output"
#define NUM_CLASSES 3
using namespace nvinfer1;


static bool ifFileExists(const char *FileName)
{
    struct stat my_stat;
    return (stat(FileName, &my_stat) == 0);
}

SampleDetector::SampleDetector()
{
    
}

void SampleDetector::loadOnnx(const std::string strModelName)
{
    Logger gLogger;
    //根据tensorrt pipeline 构建网络
    IBuilder* builder = createInferBuilder(gLogger);
    builder->setMaxBatchSize(1);
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);  
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    parser->parseFromFile(strModelName.c_str(), static_cast<int>(ILogger::Severity::kWARNING));
    IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1ULL << 30);    
    m_CudaEngine = builder->buildEngineWithConfig(*network, *config);    

    std::string strTrtName = strModelName;
    size_t sep_pos = strTrtName.find_last_of(".");
    strTrtName = strTrtName.substr(0, sep_pos) + ".trt";
    IHostMemory *gieModelStream = m_CudaEngine->serialize();
    std::string serialize_str;
    std::ofstream serialize_output_stream;
    serialize_str.resize(gieModelStream->size());   
    memcpy((void*)serialize_str.data(),gieModelStream->data(),gieModelStream->size());
    serialize_output_stream.open(strTrtName.c_str());
    serialize_output_stream<<serialize_str;
    serialize_output_stream.close();
    m_CudaContext = m_CudaEngine->createExecutionContext();
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    fprintf(stderr, "\nload ONNX model\n");
}

void SampleDetector::loadTrt(const std::string strName)
{
    Logger gLogger;
    IRuntime* runtime = createInferRuntime(gLogger);    
    initLibNvInferPlugins(&gLogger, "");
    std::ifstream fin(strName);
    std::string cached_engine = "";
    while (fin.peek() != EOF)
    { 
        std::stringstream buffer;
        buffer << fin.rdbuf();
        cached_engine.append(buffer.str());
    }
    fin.close();
    m_CudaEngine = runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
    m_CudaContext = m_CudaEngine->createExecutionContext();
    fprintf(stderr, "\nload TRT model\n");
    runtime->destroy();
}

bool SampleDetector::Init(const std::string& strModelName, float thresh)
{
    mThresh = thresh;
    std::string strTrtName = strModelName;
    size_t sep_pos = strTrtName.find_last_of(".");
    strTrtName = strTrtName.substr(0, sep_pos) + ".trt";
    fprintf(stderr, strTrtName.c_str());
    if(ifFileExists(strTrtName.c_str()))
    {        
        loadTrt(strTrtName);
    }
    else
    {
        loadOnnx(strModelName);
    }    
    // 分配输入输出的空间,DEVICE侧和HOST侧
    m_iInputIndex = m_CudaEngine->getBindingIndex(INPUT_NAME);
    // m_iOutputIndex = m_CudaEngine->getBindingIndex(OUTPUT_NAME);     

    Dims dims_i = m_CudaEngine->getBindingDimensions(m_iInputIndex);
    SDKLOG(INFO) << "input dims " << dims_i.d[0] << " " << dims_i.d[1] << " " << dims_i.d[2] << " " << dims_i.d[3];
    int size = dims_i.d[0] * dims_i.d[1] * dims_i.d[2] * dims_i.d[3];
    
    m_InputSize = cv::Size(dims_i.d[3], dims_i.d[2]);

    cudaMalloc(&m_ArrayDevMemory[m_iInputIndex], size * sizeof(float));
    m_ArrayHostMemory[m_iInputIndex] = malloc(size * sizeof(float));
    //方便NHWC到NCHW的预处理
    m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, m_ArrayHostMemory[m_iInputIndex]);
    m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, m_ArrayHostMemory[m_iInputIndex] + sizeof(float) * dims_i.d[2] * dims_i.d[3] );
    m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, m_ArrayHostMemory[m_iInputIndex] + 2 * sizeof(float) * dims_i.d[2] * dims_i.d[3]);
    m_ArraySize[m_iInputIndex] = size *sizeof(float);

    nvinfer1::ICudaEngine * & engine = m_CudaEngine;
    auto out_dims1 = engine->getBindingDimensions(engine->getBindingIndex("num_dets"));
    out_size1 = 1;
    for (int j = 0; j < out_dims1.nbDims; j++) {
      out_size1 *= out_dims1.d[j];
    }
    auto out_dims2 = engine->getBindingDimensions(engine->getBindingIndex("det_boxes"));
    out_size2 = 1;
    for (int j = 0; j < out_dims2.nbDims; j++) {
      out_size2 *= out_dims2.d[j];
    }
    auto out_dims3 = engine->getBindingDimensions(engine->getBindingIndex("det_scores"));
    out_size3 = 1;
    for (int j = 0; j < out_dims3.nbDims; j++) {
      out_size3 *= out_dims3.d[j];
    }
    auto out_dims4 = engine->getBindingDimensions(engine->getBindingIndex("det_classes"));
    out_size4 = 1;
    for (int j = 0; j < out_dims4.nbDims; j++) {
      out_size4 *= out_dims4.d[j];
    }    

    cudaError_t state;
    m_ArrayHostMemory[1] = malloc( out_size1 * sizeof(_Float32));
    state = cudaMalloc(&m_ArrayDevMemory[1], out_size1 * sizeof(_Float32));
    // state = cudaMalloc(&m_ArrayDevMemory[1], out_size1 * sizeof(int));
    m_ArraySize[1] = out_size1 *sizeof(_Float32);
    if (state) {
      std::cout << "allocate memory failed\n";
      std::abort();
    }

    m_ArrayHostMemory[2] = malloc( out_size2 * sizeof(_Float32));
    state = cudaMalloc(&m_ArrayDevMemory[2], out_size2 * sizeof(_Float32));
    m_ArraySize[2] = out_size2 *sizeof(_Float32);
    if (state) {
      std::cout << "allocate memory failed\n";
      std::abort();
    }

    m_ArrayHostMemory[3] = malloc( out_size3 * sizeof(_Float32));
    state = cudaMalloc(&m_ArrayDevMemory[3], out_size3 * sizeof(_Float32));
    m_ArraySize[3] = out_size3 *sizeof(_Float32);
    if (state) {
      std::cout << "allocate memory failed\n";
      std::abort();
    }

    m_ArrayHostMemory[4] = malloc( out_size4 * sizeof(_Float32));
    // state = cudaMalloc(&m_ArrayDevMemory[4], out_size4 * sizeof(int));
    state = cudaMalloc(&m_ArrayDevMemory[4], out_size4 * sizeof(_Float32));
    m_ArraySize[4] = out_size4 *sizeof(_Float32);
    if (state) {
      std::cout << "allocate memory failed\n";
      std::abort();
    }
    m_iClassNums = NUM_CLASSES; 

    cudaStreamCreate(&m_CudaStream);    
    m_bUninit = false;
    return  true;
    // dims_i = m_CudaEngine->getBindingDimensions(m_iOutputIndex);
    // SDKLOG(INFO) << "output dims "<< dims_i.nbDims << " " << dims_i.d[0] << " " << dims_i.d[1] << " " << dims_i.d[2];    
    // size = dims_i.d[0] * dims_i.d[1] * dims_i.d[2];
    // m_iClassNums = dims_i.d[2] - 5;
    // m_iBoxNums = dims_i.d[1];
    // cudaMalloc(&m_ArrayDevMemory[m_iOutputIndex], size * sizeof(float));
    // m_ArrayHostMemory[m_iOutputIndex] = malloc( size * sizeof(float));
    // m_ArraySize[m_iOutputIndex] = size *sizeof(float);
    // cudaStreamCreate(&m_CudaStream);    
    // m_bUninit = false;
    // return  true;
}

bool SampleDetector::UnInit()
{
    if(m_bUninit == true)
    {
        return false;
    }
    for(auto &p: m_ArrayDevMemory)
    {      
        cudaFree(p);
        p = nullptr;            
    }        
    for(auto &p: m_ArrayHostMemory)
    {        
        free(p);
        p = nullptr;        
    }        
    cudaStreamDestroy(m_CudaStream);
    m_CudaContext->destroy();
    m_CudaEngine->destroy();
    m_bUninit = true;
    return true;
}

SampleDetector::~SampleDetector()
{
    UnInit();   
}

float letterbox(
    const cv::Mat& image,
    cv::Mat& out_image,
    const cv::Size& new_shape = cv::Size(640, 384), // cv::Size (width, height)
    bool auto_flag = true,
    int stride = 32,
    const cv::Scalar& color = cv::Scalar(114, 114, 114),
    bool scale_up = true) {
    cv::Size shape = image.size();
    float r = std::min(
        (float)new_shape.height / (float)shape.height, (float)new_shape.width / (float)shape.width);
    if (!scale_up) {
      r = std::min(r, 1.0f);
    }

    int newUnpad[2]{
        (int)std::round((float)shape.width * r), (int)std::round((float)shape.height * r)};

    cv::Mat tmp;
    if (shape.width != newUnpad[0] || shape.height != newUnpad[1]) {
      cv::resize(image, tmp, cv::Size(newUnpad[0], newUnpad[1]));
    } else {
      tmp = image.clone();
    }

    float dw = new_shape.width - newUnpad[0];
    float dh = new_shape.height - newUnpad[1];

    if (auto_flag) {
      dw = (float)((int)dw % stride);
      dh = (float)((int)dh % stride);
    }

    dw /= 2.0f;
    dh /= 2.0f;

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    cv::copyMakeBorder(tmp, out_image, top, bottom, left, right, cv::BORDER_CONSTANT, color);

    return 1.0f / r;
}


bool SampleDetector::ProcessImage(const cv::Mat& img, std::vector<BoxInfo>& DetObjs, float thresh)
{
    mThresh = thresh;
    DetObjs.clear();  
    // float r = std::min(m_InputSize.height / static_cast<float>(img.rows), m_InputSize.width / static_cast<float>(img.cols));
    // cv::Size new_size = cv::Size{img.cols * r, img.rows * r};    
    // cv::Mat tmp_resized;    
    
    // cv::resize(img, tmp_resized, new_size);
    cv::Mat tmp_resized;    
    float r = letterbox(img, tmp_resized, cv::Size(640, 384), false); 
    cv::cvtColor(tmp_resized, tmp_resized, cv::COLOR_BGR2RGB);
    if(m_Resized.empty())
        m_Resized = cv::Mat( cv::Size(m_InputSize.width, m_InputSize.height), CV_8UC3, cv::Scalar(114, 114, 114));    
    
    tmp_resized.copyTo(m_Resized(cv::Rect{0, 0, tmp_resized.cols, tmp_resized.rows}));
    // cv::imwrite("tmp_resized.jpg", tmp_resized);
    // cv::imwrite("m_resized.jpg", m_Resized);
    
    m_Resized.convertTo(m_Normalized, CV_32FC3, 1/255.);
    cv::split(m_Normalized, m_InputWrappers); 

    auto ret = cudaMemcpyAsync(m_ArrayDevMemory[m_iInputIndex], m_ArrayHostMemory[m_iInputIndex], m_ArraySize[m_iInputIndex], cudaMemcpyHostToDevice, m_CudaStream);
    auto ret1 = m_CudaContext->enqueueV2(m_ArrayDevMemory, m_CudaStream, nullptr);    
    for (int i=1; i<5; i++)
    {
        ret = cudaMemcpyAsync(m_ArrayHostMemory[i], m_ArrayDevMemory[i], m_ArraySize[i], cudaMemcpyDeviceToHost, m_CudaStream);
    }
    // ret = cudaMemcpyAsync(m_ArrayHostMemory[m_iOutputIndex], m_ArrayDevMemory[m_iOutputIndex], m_ArraySize[m_iOutputIndex], cudaMemcpyDeviceToHost, m_CudaStream);
    ret = cudaStreamSynchronize(m_CudaStream);    
    float scale = r; // std::min(m_InputSize.width / (img.cols * 1.0), m_InputSize.height / (img.rows * 1.0));
    decode_outputs(m_ArrayHostMemory, mThresh, DetObjs, scale, img.cols, img.rows);
    // decode_outputs((float*)m_ArrayHostMemory[m_iOutputIndex], mThresh, DetObjs, scale, img.cols, img.rows);
    // runNms(DetObjs, 0.45);
    return true;
}

void SampleDetector::runNms(std::vector<BoxInfo>& objects, float thresh) 
{
    auto cmp_lammda = [](const BoxInfo& b1, const BoxInfo& b2){return b1.score < b2.score;};
    std::sort(objects.begin(), objects.end(), cmp_lammda);
    for(int i = 0; i < objects.size(); ++i)
    {
        if( objects[i].score < 0.1 )
        {
            continue;
        }
        for(int j = i + 1; j < objects.size(); ++j)
        {
            cv::Rect rect1 = cv::Rect{objects[i].x1, objects[i].y1, objects[i].x2 - objects[i].x1, objects[i].y2 - objects[i].y1};
            cv::Rect rect2 = cv::Rect{objects[j].x1, objects[j].y1, objects[j].x2 - objects[j].x1, objects[j].y2 - objects[j].y1};
            if(IOU(rect1, rect2) > thresh)   
            {
                objects[i].score = 0.f;
            }
        }
    }
    auto iter = objects.begin();
    while( iter != objects.end() )
    {
        if(iter->score < 0.1)
        {
            iter = objects.erase(iter);
        }
        else
        {
            ++iter;
        }
    }
}

void SampleDetector::decode_outputs(void* prob, float thresh, std::vector<BoxInfo>& objects, float scale, const int img_w, const int img_h) 
{   
    // for (int i=0;i<5;i++)
    // {
    //     int dataSize = this->m_ArraySize[i];
    //     std::cout << i << " : " << dataSize << std::endl;
    //     std::ofstream outFile("output_"+std::to_string(i)+".bin", std::ios::binary);
    //     outFile.write((char*)(((float**)this->m_ArrayHostMemory)[i]), dataSize);
    //     outFile.close();
    // }

    int32_t * num_dets = ((int32_t**)(prob))[1];
    _Float32 * det_boxes = ((_Float32**)(prob))[2];
    _Float32 * det_scores = ((_Float32**)(prob))[3]; 
    int32_t * det_classes = ((int32_t **)(prob))[4]; 
    m_iBoxNums = num_dets[0];
    int iW = m_InputSize.width;
    int iH = m_InputSize.height;
    int x_offset = (iW * scale - img_w) / 2;
    int y_offset = (iH * scale - img_h) / 2;
    for (size_t i = 0; i < int(num_dets[0]); i++) {
        float x0 = (det_boxes[i * 4]) * scale - x_offset;
        float y0 = (det_boxes[i * 4 + 1]) * scale - y_offset;
        float x1 = (det_boxes[i * 4 + 2]) * scale - x_offset;
        float y1 = (det_boxes[i * 4 + 3]) * scale - y_offset;
        // std::cout << det_scores[i] << std::endl;
        cv::Rect box(x0, y0, x1-x0, y1-y0);
        box = box &  cv::Rect(0, 0, img_w-1, img_h-1);
        // std::cout << "gagagagag:" << box.x << " "  << box.y << " " << box.width << " " << box.height << " " << det_scores[i] << " " << det_classes[i] <<std::endl;
        if(box.area() > 0) {
            BoxInfo box_info = { box.x, box.y, box.width+box.x, box.height+box.y, det_scores[i], det_classes[i]};
            objects.push_back(box_info);
        }
    // int index = i * (m_iClassNums + 5);            
    // if(prob[index + 4] > mThresh)
    // {            
    //     
    //     float x = prob[index];
    //     float y = prob[index + 1];
    //     float w = prob[index + 2];
    //     float h = prob[index + 3];            
    //     x/=scale;
    //     y/=scale;
    //     w/=scale;
    //     h/=scale;
    //     float* max_cls_pos = std::max_element(prob + index + 5, prob + index + 5 + m_iClassNums);           
    //     if((*max_cls_pos) * prob[index+4] > mThresh)
    //     {
    //         
    //         cv::Rect box{x- w / 2, y - h / 2, w, h};
    //         box = box & cv::Rect(0, 0, img_w-1, img_h-1);
    //         if( box.area() > 0)
    //         {
    //             BoxInfo box_info = { box.x, box.y, box.x + box.width, box.y + box.height, (*max_cls_pos) * prob[index+4], max_cls_pos - (prob + index + 5)};
    //             objects.push_back(box_info);
    //         }
    //     }
    // }
    }    
}
