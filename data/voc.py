import os
import xml.dom.minidom as xdom

from configuration import Config


class ParseVOC:
    def __init__(self):
        self.all_xml_dir = Config.pascal_voc_root + "Annotations"
        self.all_image_dir = Config.pascal_voc_root + "JPEGImages"
        self.pascal_voc_classes = Config.pascal_voc_classes

    @staticmethod
    def __str_to_int(x):
        return int(float(x))

    def __parse_xml(self, xml):
        """
        Parse one xml file.
        :param xml: the name of .xml file
        :return:
        """
        obj_and_box_list = []
        DOMTree = xdom.parse(os.path.join(self.all_xml_dir, xml))
        annotation = DOMTree.documentElement
        image_name = annotation.getElementsByTagName("filename")[0].childNodes[0].data
        size = annotation.getElementsByTagName("size")
        image_height = 0
        image_width = 0
        for s in size:
            image_height = s.getElementsByTagName("height")[0].childNodes[0].data
            image_width = s.getElementsByTagName("width")[0].childNodes[0].data
        obj = annotation.getElementsByTagName("object")
        for o in obj:
            o_list = []
            obj_name = o.getElementsByTagName("name")[0].childNodes[0].data
            bndbox = o.getElementsByTagName("bndbox")
            for box in bndbox:
                xmin = box.getElementsByTagName("xmin")[0].childNodes[0].data
                ymin = box.getElementsByTagName("ymin")[0].childNodes[0].data
                xmax = box.getElementsByTagName("xmax")[0].childNodes[0].data
                ymax = box.getElementsByTagName("ymax")[0].childNodes[0].data
                x_min = ParseVOC.__str_to_int(xmin)
                y_min = ParseVOC.__str_to_int(ymin)
                x_max = ParseVOC.__str_to_int(xmax)
                y_max = ParseVOC.__str_to_int(ymax)
                o_list.append(x_min)
                o_list.append(y_min)
                o_list.append(x_max)
                o_list.append(y_max)
                break
            o_list.append(self.pascal_voc_classes[obj_name])
            obj_and_box_list.append(o_list)
        return image_name, image_height, image_width, obj_and_box_list

    def __combine_info(self, image_name, image_height, image_width, box_list):
        """
        :param image_name:
        :param image_height:
        :param image_width:
        :param box_list:
        :return: string, for example: xxx.jpg image_height image_width xmin ymin xmax ymax class_type xmin ymin xmax ymax class_type ...
        """
        image_dir = self.all_image_dir + "/" + image_name
        line_str = image_dir + " " + image_height + " " + image_width + " "
        for box in box_list:
            for item in box:
                item_str = str(item)
                line_str += item_str
                line_str += " "
        line_str = line_str.strip()
        return line_str

    def write_data_to_txt(self, txt_dir):
        """
        Transform and write the VOC annotations to a txt file.
        :param txt_dir:
        :return:
        """
        with open(txt_dir, mode="a+", encoding="utf-8") as f:
            for item in os.listdir(self.all_xml_dir):
                image_name, image_height, image_width, box_list = self.__parse_xml(xml=item)
                print("Writing information of picture {} to {}".format(image_name, txt_dir))
                # Combine the information into one line.
                line_info = self.__combine_info(image_name, image_height, image_width, box_list)
                line_info += "\n"
                f.write(line_info)
