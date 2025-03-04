@echo off
cd C:\wjx\研究生\代码库\测试时间自适应\BERT_tent

python main.py -save-dir snapshot30/VLC/twitter/ -test-cover-dir ../../data/BLOCK/movie/test_data/test_cover.txt -test-stego-dir ../../data/BLOCK/movie/test_data/stego_3words_bit.txt -test True
python main.py -save-dir snapshot30/VLC/twitter/ -test-cover-dir ../../data/BLOCK/news/test_data/test_cover.txt -test-stego-dir ../../data/BLOCK/news/test_data/stego_3words_bit.txt -test True
python main.py -save-dir snapshot30/VLC/twitter/ -test-cover-dir ../../data/BLOCK/twitter/test_data/test_cover.txt -test-stego-dir ../../data/BLOCK/twitter/test_data/stego_3words_bit.txt -test True



#python main.py -save-dir snapshot30/VLC/twitter/ -test-cover-dir ../../data/VLC/movie/test_data/test_cover.txt -test-stego-dir ../../data/VLC/movie/test_data/stego_3words_bit.txt -test True
#python main.py -save-dir snapshot30/VLC/twitter/ -test-cover-dir ../../data/VLC/news/test_data/test_cover.txt -test-stego-dir ../../data/VLC/news/test_data/stego_3words_bit.txt -test True
#python main.py -save-dir snapshot30/VLC/twitter/ -test-cover-dir ../../data/VLC/twitter/test_data/test_cover.txt -test-stego-dir ../../data/VLC/twitter/test_data/stego_3words_bit.txt -test True


python main.py -save-dir snapshot30/VLC/twitter/ -test-cover-dir ../../data/ADG/movie/test_data/test_cover.txt -test-stego-dir ../../data/ADG/movie/test_data/stego_3words_bit.txt -test True
python main.py -save-dir snapshot30/VLC/twitter/ -test-cover-dir ../../data/ADG/news/test_data/test_cover.txt -test-stego-dir ../../data/ADG/news/test_data/stego_3words_bit.txt -test True
python main.py -save-dir snapshot30/VLC/twitter/ -test-cover-dir ../../data/ADG/twitter/test_data/test_cover.txt -test-stego-dir ../../data/ADG/twitter/test_data/stego_3words_bit.txt -test True

python main.py -save-dir snapshot30/VLC/twitter/ -test-cover-dir ../../data/AC/movie/test_data/test_cover.txt -test-stego-dir ../../data/AC/movie/test_data/stego_3words_bit.txt -test True
python main.py -save-dir snapshot30/VLC/twitter/ -test-cover-dir ../../data/AC/news/test_data/test_cover.txt -test-stego-dir ../../data/AC/news/test_data/stego_3words_bit.txt -test True
python main.py -save-dir snapshot30/VLC/twitter/ -test-cover-dir ../../data/AC/twitter/test_data/test_cover.txt -test-stego-dir ../../data/AC/twitter/test_data/stego_3words_bit.txt -test True








python main.py -save-dir snapshot30/VLC/twitter/ -test-cover-dir ../../data/BLOCK/movie/test_data/test_cover.txt -test-stego-dir ../../data/BLOCK/movie/test_data/stego_3words_bit.txt -test True -strategy tent
python main.py -save-dir snapshot30/VLC/twitter/ -test-cover-dir ../../data/BLOCK/news/test_data/test_cover.txt -test-stego-dir ../../data/BLOCK/news/test_data/stego_3words_bit.txt -test True -strategy tent
python main.py -save-dir snapshot30/VLC/twitter/ -test-cover-dir ../../data/BLOCK/twitter/test_data/test_cover.txt -test-stego-dir ../../data/BLOCK/twitter/test_data/stego_3words_bit.txt -test True -strategy tent



#python main.py -save-dir snapshot30/VLC/twitter/ -test-cover-dir ../../data/VLC/movie/test_data/test_cover.txt -test-stego-dir ../../data/VLC/movie/test_data/stego_3words_bit.txt -test True  -strategy tent
#python main.py -save-dir snapshot30/VLC/twitter/ -test-cover-dir ../../data/VLC/news/test_data/test_cover.txt -test-stego-dir ../../data/VLC/news/test_data/stego_3words_bit.txt -test True  -strategy tent
#python main.py -save-dir snapshot30/VLC/twitter/ -test-cover-dir ../../data/VLC/twitter/test_data/test_cover.txt -test-stego-dir ../../data/VLC/twitter/test_data/stego_3words_bit.txt -test True  -strategy tent


python main.py -save-dir snapshot30/VLC/twitter/ -test-cover-dir ../../data/ADG/movie/test_data/test_cover.txt -test-stego-dir ../../data/ADG/movie/test_data/stego_3words_bit.txt -test True  -strategy tent
python main.py -save-dir snapshot30/VLC/twitter/ -test-cover-dir ../../data/ADG/news/test_data/test_cover.txt -test-stego-dir ../../data/ADG/news/test_data/stego_3words_bit.txt -test True  -strategy tent
python main.py -save-dir snapshot30/VLC/twitter/ -test-cover-dir ../../data/ADG/twitter/test_data/test_cover.txt -test-stego-dir ../../data/ADG/twitter/test_data/stego_3words_bit.txt -test True  -strategy tent

python main.py -save-dir snapshot30/VLC/twitter/ -test-cover-dir ../../data/AC/movie/test_data/test_cover.txt -test-stego-dir ../../data/AC/movie/test_data/stego_3words_bit.txt -test True  -strategy tent
python main.py -save-dir snapshot30/VLC/twitter/ -test-cover-dir ../../data/AC/news/test_data/test_cover.txt -test-stego-dir ../../data/AC/news/test_data/stego_3words_bit.txt -test True  -strategy tent
python main.py -save-dir snapshot30/VLC/twitter/ -test-cover-dir ../../data/AC/twitter/test_data/test_cover.txt -test-stego-dir ../../data/AC/twitter/test_data/stego_3words_bit.txt -test True  -strategy tent