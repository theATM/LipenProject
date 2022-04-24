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
import math
import csv
import re

import protocol as pd
import names as NM
import random

NAMES = NM.NAMES_PL

IMAGES_PATH = "pictures/"
LABEL_FILE_PATH = "LipenLabel.csv"


def main():
    TaggerApp().run()


def binary_decomposition(x):
    p = 2 ** (int(x).bit_length() - 1)
    while int(p):
        if p & int(x):
            yield p
        p //= 2


# the Base Class of our Kivy App
class TaggerApp(App):
    def __init__(self):
        super(TaggerApp, self).__init__()
        self.image_list = self.getFiles(IMAGES_PATH)
        self.image_amount = len(self.image_list)
        random.seed(1525) #keep shuffle results the same every time
        random.shuffle(self.image_list) # change images around  (same way every time)
        self.label_file = LabelFile(LABEL_FILE_PATH,self.image_amount)
        self.image_counter = self.label_file.get_line_count() - 1
        if self.image_counter >= self.image_amount :
            print("All Images already tagged - or wrong csv file")
            exit(1)
        self.ui = None
        self.status = None
        #Window.maximize() #breaks on linux

    def getFiles(self,dir):
        image_list = []
        if dir[-1] != '/': dir = dir + '/'
        for file in os.listdir(dir):
            if os.path.isdir(os.path.join(dir, file)):
                image_list.extend(self.getFiles(dir + file))
            if os.path.isfile(os.path.join(dir, file)) and file.lower().endswith((".jpg", "jpeg", "png")):
                if dir.split('/')[0] == IMAGES_PATH or dir.split('/')[0]+'/' == IMAGES_PATH:
                    sdir = "/".join(dir.split('/')[1:]) # do not save image path to image name
                image_list.append(str(sdir + file))
        return image_list

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
        if True in class_buttons_state:
            pressed_button_index = class_buttons_state.index(True)
        #Save IT!!
        if pressed_button_index is not None:
            if self.ui.sublabels_layout_list[pressed_button_index] is not None:
                sublabel_state = [button.selected for button in self.ui.sublabels_layout_list[pressed_button_index].children][::-1]
                if True not in sublabel_state: return
                pressed_sub_state_index = sublabel_state.index(True)
            else:
                pressed_sub_state_index = 0
            extralabel_state = [button.selected for button in self.ui.extralabels_list]
            extra_labels_code = sum([(list(pd.extralabels.values())[i] if x is True else 0) for i,x in enumerate(extralabel_state)])
            self.label_file.set_tag(self.image_list[self.image_counter],pressed_button_index,pressed_sub_state_index,extra_labels_code)
            self.label_file.goto_line(direction)
        elif direction is True:
            return
        else:
            self.label_file.goto_line(direction)

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
            else: return
        self.ui.changeImage(IMAGES_PATH + image)
        #Change status:
        self.ui.changeStatus(self.image_counter+1,self.image_amount)

        #Clear Buttons
        if pressed_button_index is not None and self.ui.sublabels_layout_list[pressed_button_index] is not None:
            for button in self.ui.sublabels_layout_list[pressed_button_index].children : button.state = "normal"
        App.get_running_app().ui.sublabel_layout.clear_widgets()
        for button in self.ui.extralabels_list + self.ui.label_buttons: button.state = "normal"

        #Set Buttons if already tagged
        buttons_state = self.label_file.load_buttons_from_tag()
        if buttons_state is not None:
            class_button = self.ui.label_buttons[buttons_state[0]]
            class_button._do_press()
            class_button.on_press()
            class_button._do_release()
            if self.ui.sublabel_layout.children and len(self.ui.sublabel_layout.children[0].children) >= buttons_state[1]:
                subclass_button = self.ui.sublabel_layout.children[0].children[buttons_state[1]]
            else:
                subclass_button = None
            extra_class_buttons = []
            for decomposed in binary_decomposition(buttons_state[2]):
                index = int(math.log2(decomposed))
                extra_class_buttons.append(self.ui.extralabels_list[index])
            for button in [subclass_button] + extra_class_buttons:
                if button:
                    button._do_press()
                    button.on_press()
                    button._do_release()

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
            case (32,_) | (13,_): #Space or Enter
                button = self.next_button
                button.do_on_press()
                return

            case (8,_): #Backspace
                button = self.back_button
                button.do_on_press()
                return

            case (_,'1') | (_,'2') | (_,'3') | (_,'4') | (_,'5') | (_,'6') | (_,'7'): #Number keys
                button = self.label_buttons[int(key3) - 1]

            case (257,_)| (258,_) | (259,_) | (260,_) | (261,_) | (262,_) | (263,_): # Numpad Keys
                button = self.label_buttons[key1 - 257]

            case (_,'a') | (_,'s') | (_,'d') | (_,'f'):
                index = ['a','s','d','f'].index(key3)
                if not self.sublabel_layout.children: return
                if len(self.sublabel_layout.children[0].children) <= index: return
                button = self.sublabel_layout.children[0].children[-index-1]

            case (_,'z') | (_,'x') | (_,'c') | (_,'v') | (_,'b'):
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

class LabelFile:
    SAVE_DELAY = 10

    def __init__(self, file_path, image_amount):
        self.file_path = file_path
        self.image_amount = image_amount
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding='utf-8') as file:
                self.lines = file.readlines()
            self.current_line_index = self.get_line_count()
        else:
            self.lines = []
            self.lines.append("Name;Label;Sublabel;Extra;Author;\n")
            self.current_line_index = 1
        self.changesBeforeSave = self.SAVE_DELAY

    def save_to_file(self):
        with open(self.file_path, "w", encoding='utf-8') as file:
            file.writelines(self.lines)

    def get_line_count(self):
        return len(self.lines)

    def goto_line(self, direction):
        if direction is True and self.current_line_index < self.image_amount:
            self.current_line_index += 1
        elif direction is False and self.current_line_index > 1:
            self.current_line_index -= 1

    def set_tag(self, img_name, label_index, sublabel_index, extralabel_code):
        if self.current_line_index >= self.get_line_count():
            self.lines.append("")
        author = ""
        if len(img_name.split("_")[-1].split(".")[0]) == 3: #There is an author tag in the image name
            author = img_name.split("_")[-1].split(".")[0]
        self.lines[self.current_line_index] = ';'.join((img_name, str(label_index),
                                                        str(sublabel_index), str(extralabel_code), author)) + ";\n"
        self.update_save_delay()

    def load_buttons_from_tag(self):
        if self.current_line_index < self.get_line_count():
            tag_str = self.lines[self.current_line_index].split(';')
            return int(tag_str[1]), int(tag_str[2]), int(tag_str[3])

    def update_save_delay(self):
        self.changesBeforeSave -= 1
        if self.changesBeforeSave == 0 or self.current_line_index >= self.image_amount:
            self.save_to_file()
            self.changesBeforeSave = self.SAVE_DELAY

    def __del__(self):
        self.save_to_file()



if __name__ == '__main__':
    main()

