cd build && \
rm * -rf && \
cmake ../ && make && \
cd - && cd test/build && \
rm * -rf && \
cmake ../ && make && \
#./test-ji-api -f 1 -i ../../data/persons.jpg  -o output.jpg
#./test-ji-api -f 1 -i ../../data/zidane.jpg  -o zidane.jpg
cd -

#cd test/build && \
#./test-ji-api -f 1 -i ../../data/zidane.jpg  -o zidane.jpg && \
#cd -
