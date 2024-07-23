cmake_minimum_required(VERSION 3.5)
project(test-ji-api)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INSTALL_PREFIX "/usr/local/ev_sdk")
SET(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")


# 使用pkg-config查找必要的依赖包
find_package(OpenCV REQUIRED)
find_library(glog REQUIRED)
set(ENV{PKG_CONFIG_PATH} "ENV{PKG_CONFIG_PATH}")
find_package(PkgConfig REQUIRED)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:/home/lvsolo/git/www/ev_sdk_demo4.0_pedestrian_intrusion_yolov5/lib")

add_executable(${PROJECT_NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Algo.cpp
        )

target_include_directories(${PROJECT_NAME}
        PUBLIC ${OpenCV_INCLUDE_DIR}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
        PUBLIC "/home/lvsolo/git/www/ev_sdk_demo4.0_pedestrian_intrusion_yolov5/include/"
        )

target_link_libraries(${PROJECT_NAME}
        #/usr/local/ev_sdk/lib/libji.so
        /home/lvsolo/git/www/ev_sdk_demo4.0_pedestrian_intrusion_yolov5/build/libji.so
        ${OpenCV_LIBS}
        glog
        pthread
        )

install(TARGETS ${PROJECT_NAME} DESTINATION bin)
