#Min Python (3.10 - 3.10.2 used)
#If one wish to use python 3.7-3.9 then it is advised to change match: case to if elif (as this feature is available since python 3.10)
#If one wish to go bellow that I use dictionaries which are ordered since 3.6(?)

# Grafical Library
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.core.window import Window

import os
import csv
import re

import protocol as pd
import names as NM

NAMES = NM.NAMES_PL

IMAGES_PATH = "pictures/"
LABEL_FILE_PATH = "LipenLabel.csv"


def main():
    TaggerApp().run()


# the Base Class of our Kivy App
class TaggerApp(App):
    def __init__(self):
        super(TaggerApp, self).__init__()
        self.image_list = []
        for file in os.listdir(IMAGES_PATH):
            if os.path.isfile(os.path.join(IMAGES_PATH,file)) and file.lower().endswith((".jpg","jpeg","png")) :
                self.image_list.append(file)
        self.image_amount = len(self.image_list)
        self.label_file = LabelFile(LABEL_FILE_PATH,self.image_amount)
        self.image_counter = self.label_file.getLineCount() - 1
        if self.image_counter >= self.image_amount :
            print("All Images already tagged - or wrong csv file")
            exit(1)
        self.ui = None
        self.status = None

    def build(self):
        self.ui = TaggerLayout(cols=3, rows=1)
        Window.bind(on_key_down=self.ui.key_action)
        #Default Image
        self.ui.changeImage(IMAGES_PATH + self.image_list[self.image_counter])
        self.ui.changeStatus(self.image_counter+1,self.image_amount)
        return self.ui

    def update(self,direction=True):
        #Check which if any button pressed
        pressed_button_index = None
        class_buttons_state = [button.selected for button in self.ui.label_buttons]
        if True in  class_buttons_state:
            pressed_button_index = class_buttons_state.index(True)
        else:
            return
        #Clear Buttons
        for button in self.ui.extralabels_list: button.state = "normal"
        #Save IT!!
        if direction is True:
            if self.ui.sublabels_layout_list[pressed_button_index] is not None:
                sublabel_state = [button.selected for button in self.ui.sublabels_layout_list[pressed_button_index].children]
                if True not in sublabel_state: return
                pressed_sub_state_index = sublabel_state.index(True)
            else:
                pressed_sub_state_index = 0
            extralabel_state = [button.selected for button in self.ui.extralabels_list]
            extra_labels_code = sum([(list(pd.extralabels.values())[i] if x is True else 0) for i,x in enumerate(extralabel_state)])
            self.label_file.saveImg(self.image_list[self.image_counter],pressed_button_index,pressed_sub_state_index,extra_labels_code)

        #Change current image
        image = ""
        if direction is True:
            #Go to Next Image
            if self.image_counter + 1 < self.image_amount:
                self.image_counter += 1
                image = self.image_list[self.image_counter]
            else: return
        else:
            if self.image_counter > 0:
                self.image_counter -= 1
                image = self.image_list[self.image_counter]
                self.label_file.backLine()
            else: return
        self.ui.changeImage(IMAGES_PATH + image)
        #Change status:
        self.ui.changeStatus(self.image_counter+1,self.image_amount)



class TaggerLayout(GridLayout):
    def __init__(self, **var_args):
        super(TaggerLayout, self).__init__(**var_args)

        #Main Layout parts
        self.left_layout = GridLayout(size_hint=(.5, 1), cols=1,rows=2)
        self.image_layout = GridLayout(size_hint=(2, 1), cols=1, rows=3, padding=(0, 0))
        self.right_layout = GridLayout(size_hint=(.5, 1), cols=1)

        #add main sub layouts to layout
        self.add_widget(self.left_layout)
        self.add_widget(self.image_layout)
        self.add_widget(self.right_layout)

        #Image Layout:
        self.status = Label(text=NAMES["status"] + " 0/0",size_hint=(1, 0.1))
        self.image_layout.add_widget(self.status)

        self.img = Image(size_hint=(1, 1.75),
                        allow_stretch=True,keep_ratio=True,opacity=1)
        self.image_layout.add_widget(self.img)

        #Add some buttons at the bottom
        self.back_button = ChangeButton(forward=False,text=NAMES["back"],size_hint=(0.5, 1))
        self.next_button = ChangeButton(forward=True,text=NAMES["next"], size_hint=(1.5, 1))
        img_footer_layout = GridLayout(size_hint=(1, 0.15), cols=2, rows=1, padding=(2, 1))
        img_footer_layout.add_widget(self.back_button)
        img_footer_layout.add_widget(self.next_button)
        self.image_layout.add_widget(img_footer_layout)


        #Right Layout - All Classes
        self.label_buttons = []
        for label in pd.labels:
            new_button = LabelToggleButton(pd.labels[label],text=NAMES[label],group="class", background_color =(0, 0, 1, 1))
            self.label_buttons.append(new_button)
            self.right_layout.add_widget(new_button)


        # Left Layout
        self.sublabel_layout = BoxLayout(size_hint=(1, 1), orientation = 'vertical')
        self.extralabel_layout = BoxLayout(size_hint=(1, 1), orientation = 'vertical')
        self.left_layout.add_widget(self.sublabel_layout)
        self.left_layout.add_widget(self.extralabel_layout)

        #Sub Labels:
        self.sublabels_layout_list = []
        for sublabel in pd.sublabels:
            if pd.sublabels[sublabel] is None :
                self.sublabels_layout_list.append(None)
            else:
                new_layout = BoxLayout(size_hint=(1, 1), orientation = 'vertical')
                for sub in pd.sublabels[sublabel]:
                    new_button = SubLabelToggleButton(list(pd.sublabels.keys()).index(sublabel), pd.sublabels[sublabel][sub], text=NAMES[sub], group=sublabel, background_color =(0.1, 0.4, 0.7, 1))
                    new_layout.add_widget(new_button)
                self.sublabels_layout_list.append(new_layout)


        #Extra Labels:
        self.extralabels_list = []
        for extra in pd.extralabels:
            if extra == "normal": continue
            new_button = ExtraLabelToggleButton(pd.extralabels[extra], text=NAMES[extra], background_color =(0, 1, 0, 1))
            self.extralabels_list.append(new_button)
            self.extralabel_layout.add_widget(new_button)



    def changeImage(self,image_path):
        self.img.source = image_path

    def changeStatus(self,image_nr,image_amount):
        self.status.text = NAMES["status"] + " " + str(image_nr) + "/" + str(image_amount)

    def key_action(self, *args):
        button = None
        key1 = args[1]
        key3 = args[3]
        match (key1,key3):
            case (None,32) | (None,13): #Space or Enter
                button = self.next_button

            case (None,8): #Backspace
                button = self.back_button

            case (_,'1') | (_,'2') | (_,'3') | (_,'4') | (_,'5') | (_,'6') | (_,'7'):
                button = self.label_buttons[int(key1)-48 - 1]

            case (_,'a') | (_,'s') | (_,'d') | (_,'f'):
                index = ['a','s','d','f'].index(key3)
                if len(self.sublabel_layout.children) <= index: return
                button = self.sublabel_layout.children[index]

            case (_,'z') | (_,'x') | (_,'c') | (_,'v') | (_,'b') | (_,'n'):
                index = ['z' , 'x' , 'c' , 'v' , 'b' , 'n'].index(key3)
                button = self.extralabels_list[index]

        if button is None: return
        button._do_press()
        button.on_press()
        button._do_release()

class ChangeButton(Button):
    def __init__(self, forward,  **var_args):
        self.forward = forward
        super(ChangeButton, self).__init__(**var_args)
        self.bind(on_press=self.do_on_press)


    def do_on_press(self, instance=None):
        app = App.get_running_app()
        if self.forward:
            app.update(True)

        else:
            app.update(False)


class LabelToggleButton(ToggleButton):
    def __init__(self, label_id, **var_args):
        super(LabelToggleButton, self).__init__(**var_args)
        self.selected = False
        self.label_id = label_id

    def on_state(self, widget, value):
        if value == 'down':
            self.selected = True
        else:
            self.selected = False

    def _do_press(self):
        if self.state == 'normal':
            ToggleButton._do_press(self)
            self.selected = True
            self.add_sub_buttons()

    def add_sub_buttons(self):
        app = App.get_running_app()
        app.ui.sublabel_layout.clear_widgets()
        if app.ui.sublabels_layout_list[self.label_id] is not None:
            app.ui.sublabel_layout.add_widget(app.ui.sublabels_layout_list[self.label_id])


class SubLabelToggleButton(ToggleButton):
    def __init__(self, label_id ,sub_id, **var_args):
        super(SubLabelToggleButton, self).__init__(**var_args)
        self.label_id = label_id
        self.selected = False

    def on_state(self, widget, value):
        if value == 'down':
            self.selected = True
        else:
            self.selected = False

    def _do_press(self):
        if self.state == 'normal':
            ToggleButton._do_press(self)
            self.selected = True


class ExtraLabelToggleButton(ToggleButton):
    def __init__(self, extra_id, **var_args):
        super(ExtraLabelToggleButton, self).__init__(**var_args)
        self.extra_id = extra_id
        self.selected = False

    def on_state(self, widget, value):
        if value == 'down':
            self.selected = True
        else:
            self.selected = False

    def _do_press(self):
        ToggleButton._do_press(self)
        if self.state == 'normal':
            self.selected = True


class LabelFile:
    def __init__(self, file_path, image_amount):
        self.file_path = file_path
        self.file = None
        self.neg_counter = 0
        self.image_amount = image_amount
        if os.path.exists(self.file_path):
            self.file = open(self.file_path, "a+t", encoding='utf-8')
            self.file.seek(0)
            self.line_count = sum([1 for _ in enumerate(self.file)])
        else:
            self.file = open(self.file_path, "wt" , encoding='utf-8')
            #write first descriptive line:
            self.file.write("Name;Label;Sublabel;Extra;Author\n")
            self.line_count = 1
        self.file.close()

    def getLineCount(self):
        return self.line_count

    def saveImg(self,img,label_index,sublabel_index,extralabel_code):
        if self.line_count  >= self.image_amount and self.neg_counter <= 0:
            return
        #decode
        author = ""
        if str(img)[3] == '_':
            author = img[0:3]
        save_string = str(img) + ";" + str(label_index) + ";" + str(sublabel_index) + ";" + str(extralabel_code) + ";" + str(author) + ";\n"
        #save to file
        if self.neg_counter > 0:
            newlinebundle = self.findLastNewLine(self.neg_counter +1)
            if newlinebundle is None:
                self.neg_counter += 1
                return
            new_pos, _ = newlinebundle
        with open(self.file_path, "r+t") as self.file:
            self.file.seek(0, os.SEEK_END)
            self.line_count += 1
            if self.neg_counter > 0:
                self.file.seek(new_pos, 0)
                self.neg_counter-=1
                self.line_count-=1
            self.file.write(save_string)



    def backLine(self):
        self.neg_counter+=1

    def findLastNewLine(self,lines_back=1):
        with open(self.file_path, "rb") as self.file:
            self.file.seek(0,os.SEEK_END)
            end_pos = self.file.tell()
            try:
                for _ in range(0,lines_back):
                    self.file.seek(-2, os.SEEK_CUR)
                    while self.file.read(1) != b'\n':
                        self.file.seek(-2, os.SEEK_CUR)
                last_pos = self.file.tell()
            except OSError:
                return None # catch OSError in case of a one line file
            return (last_pos, end_pos)


    def __del__(self):
        if self.file is not None:
            self.file.close()
            self.file = None



if __name__ == '__main__':
    main()

