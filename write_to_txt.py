from data.voc import ParseVOC
from configuration import Config

if __name__ == '__main__':
    parse_voc = ParseVOC()
    parse_voc.write_data_to_txt(txt_dir=Config.txt_file_dir)