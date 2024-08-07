#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jordy Thielen (jordy.thielen@donders.ru.nl)

Python implementation of a keyboard for the noise-tagging project.
"""
import json
import random

import numpy as np
from pylsl import StreamInfo, StreamOutlet
from psychopy import visual, event, monitors, misc
from speller.utils.logging import logger
from dareplane_utils.stream_watcher.lsl_stream_watcher import StreamWatcher
import pkg_resources
from symspellpy import SymSpell, Verbosity

STREAM = True
SCREEN = 0
SCREEN_SIZE = (1920, 1080)  # Mac: (1792, 1120), LabPC: (1920, 1080)
SCREEN_WIDTH = 53.0  # Mac: (34,5), LabPC: 53.0
SCREEN_DISTANCE = 60.0
SCREEN_COLOR = (0, 0, 0)
FR = 60  # screen frame rate
PR = 60  # codes presentation rate

STT_WIDTH = 2.2
STT_HEIGHT = 2.2

TEXT_FIELD_HEIGHT = 3.0

KEY_WIDTH = 3.0
KEY_HEIGHT = 3.0
KEY_SPACE = 0.5
KEY_COLORS = ["black", "white", "green", "blue"]
KEYSABCDE = [
    ["A", "B", "C", "D", "E", "F", "G", "H"],
    ["I", "J", "K", "L", "M", "N", "O", "P"],
    ["Q", "R", "S", "T", "U", "V", "W", "X"],
    ["Y", "Z", "underscore", "dot", "question", "exclamation", "smaller", "hash"]]

KEYSQWERTY = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "smaller"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["Z", "X", "C", "V", "B", "N", "M", "dot", "question", "exclamation", "hash"],
    ["underscore"]]

SPECIAL_CHARACTERS = {
    "smaller": "<",
    "dot": ".",
    "question": "?",
    "exclamation": "!",
    "hash": "#",
    "underscore": " "
}

CUE_TIME = 0.8
TRIAL_TIME = 4.2
FEEDBACK_TIME = 0.5
ITI_TIME = 0.5
CODE = "mgold_61_6521"


def flatten(list):
    return [x for xs in list for x in xs]

class Keyboard(object):
    """
    A keyboard with keys and text fields.
    """

    def __init__(self, size, width, distance, screen=0, window_color=(0, 0, 0), stream=True):
        """
        Create a keyboard.

        Args:
            size (array-like):
                The (width, height) of the window in pixels, i.e., resolution
            width (float):
                The width of the screen in centimeters
            distance (float):
                The distance of the user to the screen in centimeters
            screen (int):
                The screen number that is used, default: 0
            window_color (array-like):
                The background color of the window, default: (0, 0, 0)
            stream (bool):
                Whether or not to log events/markers in an LSL stream. Default: True
        """
        # Set up monitor (sets pixels per degree)
        self.monitor = monitors.Monitor("testMonitor", width=width, distance=distance)
        self.monitor.setSizePix(size)

        # Set up window
        self.window = visual.Window(monitor=self.monitor, screen=screen, units="pix", size=size, color=window_color, fullscr=True, waitBlanking=False, allowGUI=False)
        self.window.setMouseVisible(False)

        # Initialize keys and fields
        self.keys = dict()
        self.fields = dict()

        # Setup LSL stream
        self.stream = stream
        if self.stream:
            self.outlet = StreamOutlet(StreamInfo(name='KeyboardMarkerStream', type='Markers', channel_count=1, nominal_srate=0, channel_format='string', source_id='KeyboardMarkerStream'))

    def get_size(self):
        """
        Get the size of the window in pixels, i.e., resolution.

        Returns:
            (array-like):
                The (width, height) of the window in pixels, i.e., resolution
        """
        return self.window.size

    def get_pixels_per_degree(self):
        """
        Get the pixels per degree of visual angle of the window.

        Returns:
            (float):
                The pixels per degree of visual angle
        """
        return misc.deg2pix(1.0, self.monitor)

    def get_framerate(self):
        """
        Get the framerate in Hz of the window.

        Returns:
            (float):
                The framerate in Hz
        """
        return int(np.round(self.window.getActualFrameRate()))

    def add_key(self, name, size, pos, images=["black.png", "white.png"]):
        """
        Add a key to the keyboard.

        Args:
            name (str):
                The name of the key, if none then text is used
            size (array-like):
                The (width, height) of the key in pixels
            pos (array-like):
                The (x, y) coordinate of the center of the key, relative to the center of the window
            images (array-like):
                The images of the key. The first image is the default key. Indices will correspond to the
                values of the codes. Default: ["black.png", "white.png"]
        """
        assert name not in self.keys, "Trying to add a box with a name that already exists!"
        self.keys[name] = []
        for image in images:
            self.keys[name].append(visual.ImageStim(win=self.window, image=image,
                                                    units="pix", pos=pos, size=size, autoLog=False, name=name))

        # Set autoDraw to True for first default key to keep app visible
        self.keys[name][0].setAutoDraw(True)

    def add_text_field(self, name, text, size, pos, field_color=(0, 0, 0), text_color=(-1, -1, -1)):
        """
        Add a text field to the keyboard.

        Args:
            name (str):
                The name of the text field, if none then text is used
            text (str):
                The text on the text field
            size (array-like):
                The (width, height) of the text field in pixels
            pos (array-like):
                The (x, y) coordinate of the center of the text field, relative to the center of the window
            field_color (array-like):
                The color of the background of the text field, default: (0, 0, 0)
            text_color (array-like):
                The color of the text on the text field, default: (-1, -1, -1)
        """
        assert name not in self.fields, "Trying to add a text field with a name that already extists!"
        self.fields[name] = self.fields[name] = visual.TextBox2(win=self.window, text=text, font='Courier',
                                                                units="pix", pos=pos, size=size, letterHeight=0.5*size[1],
                                                                color=text_color, fillColor=field_color, alignment="left",
                                                                autoDraw=True, autoLog=False)

    def set_field_text(self, name, text):
        """
        Set the text of a text field.

        Args:
            name (str):
                The name of the key
            text (str):
                The text
        """
        self.fields[name].setText(text)
        self.window.flip()

    def log(self, marker, on_flip=False):
        if self.stream and marker is not None:
            if not isinstance(marker, list):
                marker = [marker]
            if on_flip:
                self.window.callOnFlip(self.outlet.push_sample, marker)
            else:
                self.outlet.push_sample(marker)

    def run(self, codes, duration=None, start_marker=None, stop_marker=None):
        """
        Present a trial with concurrent flashing of each of the symbols.

        Args:
            codes (dict):
                A dictionary with keys being the symbols to flash and the value a list (the code
                sequence) of integer states (images) for each frame
            duration (float):
                The duration of the trial in seconds. If the duration is longer than the code
                sequence, it is repeated. If no duration is given, the full length of the first
                code is used. Default: None
        """
        # Set number of frames
        if duration is None:
            n_frames = len(codes[list(codes.keys())[0]])
        else:
            n_frames = int(duration * self.get_framerate())

        # Set autoDraw to False for full control
        for key in self.keys.values():
            key[0].setAutoDraw(False)

        # Send start marker
        self.log(start_marker, on_flip=True)

        # Loop frame flips
        for i in range(n_frames):

            # Check quiting
            if i % 60 == 0:
                if self.is_quit():
                    self.quit()

            # Draw keys with color depending on code state
            for name, code in codes.items():
                self.keys[name][code[i % len(code)]].draw()
            self.window.flip()

        # Send stop markers
        self.log(stop_marker)

        # Set autoDraw to True to keep app visible
        for key in self.keys.values():
            key[0].setAutoDraw(True)
        self.window.flip()

    def is_quit(self):
        """
        Test if a quit is forced by the user by a key-press.

        Returns:
            (bool):
                True is quit forced, otherwise False
        """
        # If quit keys pressed, return True
        if len(event.getKeys(keyList=["q", "escape"])) > 0:
            return True
        return False

    def quit(self):
        """
        Quit the keyboard.
        """
        self.window.setMouseVisible(True)
        self.window.close()
        # core.quit()

def training(code=CODE, layout_qwerty = False):
    """
    Example experiment with initial setup and highlighting and presenting a few trials.
    """
    logger.setLevel(10)

    keys: np.array
    if layout_qwerty:
        keys = KEYSQWERTY
    else:
        keys = KEYSABCDE
    # Initialize keyboard
    keyboard = Keyboard(size=SCREEN_SIZE, width=SCREEN_WIDTH, distance=SCREEN_DISTANCE, screen=SCREEN, window_color=SCREEN_COLOR, stream=STREAM)
    ppd = keyboard.get_pixels_per_degree()

    # Add stimulus timing tracker at left top of the screen
    x_pos = -SCREEN_SIZE[0] / 2 + STT_WIDTH / 2 * ppd
    y_pos = SCREEN_SIZE[1] / 2 - STT_HEIGHT / 2 * ppd
    images = ["images/black.png", "images/white.png"]
    keyboard.add_key("stt", (STT_WIDTH * ppd, STT_HEIGHT * ppd), (x_pos, y_pos), images)

    # Add text field at the top of the screen
    x_pos = STT_WIDTH * ppd
    y_pos = SCREEN_SIZE[1] / 2 - TEXT_FIELD_HEIGHT * ppd / 2
    keyboard.add_text_field("text", "", (SCREEN_SIZE[0] - STT_WIDTH * ppd, TEXT_FIELD_HEIGHT * ppd), (x_pos, y_pos), (0, 0, 0), (-1, -1, -1))


    # Add the keys
    for y in range(len(keys)):
        for x in range(len(keys[y])):
            x_pos = (x - len(keys[y]) / 2 + 0.5) * (KEY_WIDTH + KEY_SPACE) * ppd
            y_pos = -(y - len(keys) / 2) * (KEY_HEIGHT + KEY_SPACE) * ppd - TEXT_FIELD_HEIGHT * ppd
            images = [f"images/{keys[y][x]}_{color}.png" for color in KEY_COLORS]
            keyboard.add_key(keys[y][x], (KEY_WIDTH * ppd, KEY_HEIGHT * ppd), (x_pos, y_pos), images)

    # Load sequences
    if code != "onoff":
        tmp = np.load(f"D:/Users/bci/bachelor_project_s1028931/ThesisSpellerProject/dp-speller/speller/codes/{code}.npz")["codes"]
    codes = dict()
    i = 0
    for row in keys:
        for key in row:
            if code == "onoff":
                codes[key] = [1, 0]
            else:
                codes[key] = tmp[:, i].tolist()
            i += 1
    if code == "onoff":
        codes["stt"] = [1, 0]
    else:
        codes["stt"] = [1] + [0] * int((1 + TRIAL_TIME) * keyboard.get_framerate())

    # Set highlights
    highlights = dict()
    for row in keys:
        for key in row:
            highlights[key] = [0]
    highlights["stt"] = [0]



    # Wait for start
    keyboard.window.setMouseVisible(True)
    keyboard.set_field_text("text", "Press button to start.")
    logger.info("Press button to start.")
    event.waitKeys()
    keyboard.set_field_text("text", "")
    logger.info("Starting.")

    # Log codes
    keyboard.log([json.dumps({"codes":codes})])

    # Start experiment
    keyboard.log(marker=["start_experiment"])
    keyboard.set_field_text("text", "Starting...")
    keyboard.run(highlights, 5.0)
    keyboard.set_field_text("text", "")

    # Loop trials
    text = ""
    rand_order = list(range(0, len(flatten(keys))))
    random.shuffle(rand_order)
    i_trial = 0

    while len(rand_order) > 0:

        logger.debug("number of remaining targets:" + str(len(rand_order)))
        # Set target
        target = rand_order.pop()
        logger.debug("target = " + str(target))

        row_index = 0
        target_ = target
        while target_ >= len(keys[row_index]):
            target_ -= len(keys[row_index])
            row_index += 1
        target_key = keys[row_index][target_]
        logger.info(f"{1 + i_trial:03d}/{len(flatten(keys))}\t{target_key}\t{target}")

        keyboard.log([json.dumps({"target":target})])
        keyboard.log([json.dumps({"target key":target_key})])

        # Cue
        highlights[target_key] = [-2]
        keyboard.run(highlights, CUE_TIME,
                     start_marker=["start_cue"],
                     stop_marker=["stop_cue"])
        highlights[target_key] = [0]

        # Trial
        keyboard.run(codes, TRIAL_TIME,
                     start_marker=["start_trial"],
                     stop_marker=["stop_trial"])


        # Inter-trial time
        keyboard.run(highlights, ITI_TIME,
                     start_marker=["start_intertrial"],
                     stop_marker=["stop_intertrial"])

        i_trial += 1


    # Stop experiment

    keyboard.log(marker=["stop_experiment"])
    keyboard.window.setMouseVisible(True)
    keyboard.set_field_text("text", "Experiment finished. Press button to close.")
    logger.info("Experiment finished. Press button to close.")
    event.waitKeys()
    keyboard.set_field_text("text", "Closing...")
    logger.info("Closing.")
    keyboard.run(highlights, 5.0)
    keyboard.set_field_text("text", "")
    keyboard.quit()
    logger.info("Experiment closed.")

    return 0

def online(n_trials = 10, code=CODE, layout_qwerty = False):
    """
    Example experiment with initial setup and highlighting and presenting a few trials.
    """
    logger.setLevel(10)

    keys: np.array
    if layout_qwerty:
        keys = KEYSQWERTY
    else:
        keys = KEYSABCDE
    # Initialize keyboard
    keyboard = Keyboard(size=SCREEN_SIZE, width=SCREEN_WIDTH, distance=SCREEN_DISTANCE, screen=SCREEN, window_color=SCREEN_COLOR, stream=STREAM)
    ppd = keyboard.get_pixels_per_degree()

    # Add stimulus timing tracker at left top of the screen
    x_pos = -SCREEN_SIZE[0] / 2 + STT_WIDTH / 2 * ppd
    y_pos = SCREEN_SIZE[1] / 2 - STT_HEIGHT / 2 * ppd
    images = ["images/black.png", "images/white.png"]
    keyboard.add_key("stt", (STT_WIDTH * ppd, STT_HEIGHT * ppd), (x_pos, y_pos), images)

    # Add text field at the top of the screen
    x_pos = STT_WIDTH * ppd
    y_pos = SCREEN_SIZE[1] / 2 - TEXT_FIELD_HEIGHT * ppd / 2
    keyboard.add_text_field("text", "", (SCREEN_SIZE[0] - STT_WIDTH * ppd, TEXT_FIELD_HEIGHT * ppd), (x_pos, y_pos), (0, 0, 0), (-1, -1, -1))

    # Add the keys
    for y in range(len(keys)):
        for x in range(len(keys[y])):
            x_pos = (x - len(keys[y]) / 2 + 0.5) * (KEY_WIDTH + KEY_SPACE) * ppd
            y_pos = -(y - len(keys) / 2) * (KEY_HEIGHT + KEY_SPACE) * ppd - TEXT_FIELD_HEIGHT * ppd
            images = [f"images/{keys[y][x]}_{color}.png" for color in KEY_COLORS]
            keyboard.add_key(keys[y][x], (KEY_WIDTH * ppd, KEY_HEIGHT * ppd), (x_pos, y_pos), images)

    # Load sequences
    if code != "onoff":
        tmp = np.load(f"D:/Users/bci/bachelor_project_s1028931/ThesisSpellerProject/dp-speller/speller/codes/{code}.npz")["codes"]
    codes = dict()
    i = 0
    for row in keys:
        for key in row:
            if code == "onoff":
                codes[key] = [1, 0]
            else:
                codes[key] = tmp[:, i].tolist()
            i += 1
    if code == "onoff":
        codes["stt"] = [1, 0]
    else:
        codes["stt"] = [1] + [0] * int((1 + TRIAL_TIME) * keyboard.get_framerate())

    # Set highlights
    highlights = dict()
    for row in keys:
        for key in row:
            highlights[key] = [0]
    highlights["stt"] = [0]



    # Wait for start
    keyboard.window.setMouseVisible(True)
    keyboard.set_field_text("text", "Press button to start.")
    logger.info("Press button to start.")
    event.waitKeys()
    keyboard.set_field_text("text", "")
    logger.info("Starting.")


    # Log codes
    keyboard.log([json.dumps({"codes":codes})])

    #connect to decoder stream
    sw = StreamWatcher(name="decoder")
    sw.connect_to_stream()

    # Start experiment
    keyboard.log(marker=["start_experiment"])
    keyboard.set_field_text("text", "Starting...")
    keyboard.run(highlights, 5.0)
    keyboard.set_field_text("text", "")

    text = ""
    finished = False
    finish_check = False
    decoder_data_old = []
    i = 0
    while not finished:
        i += 1
        logger.debug(f"Starting trial {i}")

        keyboard.run(codes, TRIAL_TIME,
                     start_marker=["start_trial"],
                     stop_marker=["stop_trial"])

        decode_result = []

        while len(decode_result) == 0:
            sw.update()
            decoder_data_raw = sw.unfold_buffer()
            decoder_data = decoder_data_raw[decoder_data_raw != 0]
            if len(decoder_data) > len(decoder_data_old):
                decode_result = decoder_data[len(decoder_data_old):]

        logger.debug(f"Received {decode_result} from decoder stream")

        decoder_data_old = decoder_data

        decode_result[0] -= 1
        row_index = 0
        target_ = decode_result[0]
        while target_ >= len(keys[row_index]):
            target_ -= len(keys[row_index])
            row_index += 1
        target_key = keys[row_index][target_]
        logger.debug(f"{decode_result} corresponds to {target_key}")

        if target_key == "hash":
            highlights[target_key] = [3]
            keyboard.run(highlights, FEEDBACK_TIME,
                         start_marker=["start_feedback"],
                         stop_marker=["stop_feedback"])
            highlights[target_key] = [0]
            if finish_check:
                finished = True
            else:
                finish_check = True
            logger.debug(f"end of trial {i}")
            break

        if target_key == "smaller":
            text = text[:-1]
            highlights[target_key] = [3]
            keyboard.run(highlights, FEEDBACK_TIME,
                         start_marker=["start_feedback"],
                         stop_marker=["stop_feedback"])
            highlights[target_key] = [0]
            keyboard.set_field_text("text", text)
            logger.debug(f"end of trial {i}")
            break

        if len(target_key) <= 1:
            character = target_key
        else:
            character = SPECIAL_CHARACTERS.get(target_key)

        text += character
        keyboard.set_field_text("text", text)

        highlights[target_key] = [3]
        keyboard.run(highlights, FEEDBACK_TIME,
                     start_marker=["start_feedback"],
                     stop_marker=["stop_feedback"])
        highlights[target_key] = [0]


        # Inter-trial time
        keyboard.run(highlights, ITI_TIME,
                     start_marker=["start_intertrial"],
                     stop_marker=["stop_intertrial"])


        logger.debug(f"end of trial {i}")

    logger.debug(f"final text is \'{text}\'")

    # Code for clicking keys
    # mouse = event.Mouse(visible=False)
    # text = ""
    #
    # i = 0
    # while i < n_trials:
    #     key_pressed = False
    #     mouse.setVisible(1)
    #     while not mouse.getPressed()[0]:
    #         pass
    #     x_mouse, y_mouse = mouse.getPos()
    #     mouse.setVisible(0)
    #     for key in keyboard.keys.values():
    #         x_key, y_key = key[0].pos
    #         size = key[0].size[0]
    #         x_left = x_key - size
    #         x_right = x_key + size
    #         y_top = y_key + size
    #         y_bottom = y_key - size
    #         if x_left <= x_mouse <= x_right and y_bottom <= y_mouse <= y_top:
    #             logger.info(key[0].name + ' pressed!')
    #             key_pressed = True
    #
    #             #TODO: change into actual backspace key
    #             if key[0].name == "smaller":
    #                 text = text[:-1]
    #                 highlights[key[0].name] = [3]
    #                 keyboard.run(highlights, FEEDBACK_TIME,
    #                              start_marker=["start_feedback"],
    #                              stop_marker=["stop_feedback"])
    #                 highlights[key[0].name] = [0]
    #                 keyboard.set_field_text("text", text)
    #                 i -= 1
    #                 break
    #
    #             if len(key[0].name) <= 1:
    #                 character = key[0].name
    #             else:
    #                 character = SPECIAL_CHARACTERS.get(key[0].name)
    #
    #             text += character
    #
    #             highlights[key[0].name] = [3]
    #             keyboard.run(highlights, FEEDBACK_TIME,
    #                          start_marker=["start_feedback"],
    #                          stop_marker=["stop_feedback"])
    #             highlights[key[0].name] = [0]
    #             keyboard.set_field_text("text", text)
    #             i += 1
    #             break
    #
    #     if not key_pressed:
    #         logger.info('no key pressed')


    # Stop experiment

    keyboard.log(marker=["stop_experiment"])
    keyboard.log(marker=[f"Number of required trials: {i}"])
    keyboard.set_field_text("text", "Experiment finished. Press button to close.")
    logger.info("Experiment finished. Press button to close.")
    event.waitKeys()
    keyboard.set_field_text("text", "Closing...")
    logger.info("Closing.")
    keyboard.run(highlights, 5.0)
    keyboard.set_field_text("text", "")
    keyboard.quit()
    logger.info("Experiment closed.")

    return 0

def online_autocomplete(n_trials = 10, code=CODE, layout_qwerty = False):
    """
    Example experiment with initial setup and highlighting and presenting a few trials.
    """
    logger.setLevel(10)

    keys: np.array
    if layout_qwerty:
        keys = KEYSQWERTY
    else:
        keys = KEYSABCDE
    # Initialize keyboard
    keyboard = Keyboard(size=SCREEN_SIZE, width=SCREEN_WIDTH, distance=SCREEN_DISTANCE, screen=SCREEN, window_color=SCREEN_COLOR, stream=STREAM)
    ppd = keyboard.get_pixels_per_degree()

    # Add stimulus timing tracker at left top of the screen
    x_pos = -SCREEN_SIZE[0] / 2 + STT_WIDTH / 2 * ppd
    y_pos = SCREEN_SIZE[1] / 2 - STT_HEIGHT / 2 * ppd
    images = ["images/black.png", "images/white.png"]
    keyboard.add_key("stt", (STT_WIDTH * ppd, STT_HEIGHT * ppd), (x_pos, y_pos), images)

    # Add text field at the top of the screen
    x_pos = STT_WIDTH * ppd
    y_pos = SCREEN_SIZE[1] / 2 - TEXT_FIELD_HEIGHT * ppd / 2
    keyboard.add_text_field("text", "", (SCREEN_SIZE[0] - STT_WIDTH * ppd, TEXT_FIELD_HEIGHT * ppd), (x_pos, y_pos), (0, 0, 0), (-1, -1, -1))

    keyboard.add_text_field("suggestion", "", (SCREEN_SIZE[0] - STT_WIDTH * ppd, TEXT_FIELD_HEIGHT * ppd),
                            (x_pos, y_pos - TEXT_FIELD_HEIGHT*ppd), (0.1, 0.1, 0.1), (-1, -1, -1))
    # Add the keys
    for y in range(len(keys)):
        for x in range(len(keys[y])):
            x_pos = (x - len(keys[y]) / 2 + 0.5) * (KEY_WIDTH + KEY_SPACE) * ppd
            y_pos = -(y - len(keys) / 2) * (KEY_HEIGHT + KEY_SPACE) * ppd - TEXT_FIELD_HEIGHT * ppd
            images = [f"images/{keys[y][x]}_{color}.png" for color in KEY_COLORS]
            keyboard.add_key(keys[y][x], (KEY_WIDTH * ppd, KEY_HEIGHT * ppd), (x_pos, y_pos), images)

    # Load sequences
    if code != "onoff":
        tmp = np.load(f"D:/Users/bci/bachelor_project_s1028931/ThesisSpellerProject/dp-speller/speller/codes/{code}.npz")["codes"]
    codes = dict()
    i = 0
    for row in keys:
        for key in row:
            if code == "onoff":
                codes[key] = [1, 0]
            else:
                codes[key] = tmp[:, i].tolist()
            i += 1
    if code == "onoff":
        codes["stt"] = [1, 0]
    else:
        codes["stt"] = [1] + [0] * int((1 + TRIAL_TIME) * keyboard.get_framerate())

    # Set highlights
    highlights = dict()
    for row in keys:
        for key in row:
            highlights[key] = [0]
    highlights["stt"] = [0]



    # Wait for start
    keyboard.window.setMouseVisible(True)
    keyboard.set_field_text("text", "Press button to start.")
    keyboard.set_field_text("suggestion", "")
    logger.info("Press button to start.")
    event.waitKeys()
    keyboard.set_field_text("text", "")
    logger.info("Starting.")


    # Log codes
    keyboard.log([json.dumps({"codes":codes})])

    #connect to decoder stream
    sw = StreamWatcher(name="decoder")
    sw.connect_to_stream()

    # Start experiment
    keyboard.log(marker=["start_experiment"])
    keyboard.set_field_text("text", "Starting...")
    keyboard.run(highlights, 5.0)
    keyboard.set_field_text("text", "")

    text = ""
    suggestion = ""

    sym_spell = SymSpell()
    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dictionary_path, 0, 1)
    bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
    sym_spell.load_bigram_dictionary(bigram_path, 0, 2)

    finished = False
    finish_check = False
    decoder_data_old = []
    i = 0
    while not finished:
        i += 1
        logger.debug(f"Starting trial {i}")

        keyboard.run(codes, TRIAL_TIME,
                     start_marker=["start_trial"],
                     stop_marker=["stop_trial"])

        decode_result = []

        while len(decode_result) == 0:
            sw.update()
            # decoder_data_raw = sw.unfold_buffer()
            # decoder_data = decoder_data_raw[decoder_data_raw != 0]
            # if len(decoder_data) > len(decoder_data_old):
            #     decode_result = decoder_data[len(decoder_data_old):]
            if sw.n_new > 0:
                decode_result = sw.unfold_buffer()[-sw.n_new:]
                sw.n_new = 0

        logger.debug(f"Received {decode_result} from decoder stream")

        # decoder_data_old = decoder_data

        decode_result[0] -= 1
        row_index = 0
        target_ = decode_result[0]
        while target_ >= len(keys[row_index]):
            target_ -= len(keys[row_index])
            row_index += 1
        target_key = keys[row_index][target_]
        logger.debug(f"{decode_result} corresponds to {target_key}")

        if target_key == "hash":
            highlights[target_key] = [3]
            keyboard.run(highlights, FEEDBACK_TIME,
                         start_marker=["start_feedback"],
                         stop_marker=["stop_feedback"])
            highlights[target_key] = [0]
            if finish_check:
                finished = True
            else:
                finish_check = True
                keyboard.set_field_text("suggestion", "Finished? Select '#' again to stop.")
            logger.debug(f"end of trial {i}")
            break

        if target_key == "smaller":
            text = text[:-1]
            highlights[target_key] = [3]
            keyboard.run(highlights, FEEDBACK_TIME,
                         start_marker=["start_feedback"],
                         stop_marker=["stop_feedback"])
            highlights[target_key] = [0]
            keyboard.set_field_text("text", text)
            logger.debug(f"end of trial {i}")
            break

        if target_key == "exclamation":
            text = suggestion
            highlights[target_key] = [3]
            keyboard.run(highlights, FEEDBACK_TIME,
                         start_marker=["start_feedback"],
                         stop_marker=["stop_feedback"])
            highlights[target_key] = [0]
            keyboard.set_field_text("text", text)

            suggestions = sym_spell.lookup_compound(text, max_edit_distance=2, transfer_casing=True)
            suggestion = suggestions[0].term
            keyboard.set_field_text("suggestion", suggestion)
            logger.debug(f"end of trial {i}")
            break

        if len(target_key) <= 1:
            character = target_key
        else:
            character = SPECIAL_CHARACTERS.get(target_key)

        text += character
        keyboard.set_field_text("text", text)

        highlights[target_key] = [3]
        keyboard.run(highlights, FEEDBACK_TIME,
                     start_marker=["start_feedback"],
                     stop_marker=["stop_feedback"])
        highlights[target_key] = [0]


        # Inter-trial time
        keyboard.run(highlights, ITI_TIME,
                     start_marker=["start_intertrial"],
                     stop_marker=["stop_intertrial"])
        logger.debug(f"end of trial {i}")

    logger.debug(f"final text is \'{text}\'")

    # Code for clicking keys
    # mouse = event.Mouse(visible=False)
    # text = ""
    # suggestion = ""
    #
    # sym_spell = SymSpell()
    # dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    # sym_spell.load_dictionary(dictionary_path, 0, 1)
    # bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
    # sym_spell.load_bigram_dictionary(bigram_path, 0, 2)
    #
    # finished = False
    # finish_check = False
    #
    #
    # while not finished:
    #     key_pressed = False
    #     mouse.setVisible(1)
    #     while not mouse.getPressed()[0]:
    #         pass
    #     x_mouse, y_mouse = mouse.getPos()
    #     mouse.setVisible(0)
    #     for key in keyboard.keys.values():
    #         x_key, y_key = key[0].pos
    #         size = key[0].size[0]
    #         x_left = x_key - size
    #         x_right = x_key + size
    #         y_top = y_key + size
    #         y_bottom = y_key - size
    #         if x_left <= x_mouse <= x_right and y_bottom <= y_mouse <= y_top:
    #             logger.info(key[0].name + ' pressed!')
    #             key_pressed = True
    #
    #             if key[0].name == "hash":
    #                 highlights[key[0].name] = [3]
    #                 keyboard.run(highlights, FEEDBACK_TIME,
    #                              start_marker=["start_feedback"],
    #                              stop_marker=["stop_feedback"])
    #                 highlights[key[0].name] = [0]
    #                 if finish_check:
    #                     finished = True
    #                 else:
    #                     finish_check = True
    #                     keyboard.set_field_text("suggestion", "Finished? Select '#' again to stop.")
    #                 break
    #
    #             finish_check = False
    #
    #             #TODO: change into actual backspace key
    #             if key[0].name == "smaller":
    #                 text = text[:-1]
    #                 highlights[key[0].name] = [3]
    #                 keyboard.run(highlights, FEEDBACK_TIME,
    #                              start_marker=["start_feedback"],
    #                              stop_marker=["stop_feedback"])
    #                 highlights[key[0].name] = [0]
    #                 keyboard.set_field_text("text", text)
    #                 break
    #
    #             if key[0].name == "exclamation":
    #                 text = suggestion
    #                 highlights[key[0].name] = [3]
    #                 keyboard.run(highlights, FEEDBACK_TIME,
    #                              start_marker=["start_feedback"],
    #                              stop_marker=["stop_feedback"])
    #                 highlights[key[0].name] = [0]
    #                 keyboard.set_field_text("text", text)
    #
    #                 suggestions = sym_spell.lookup_compound(text, max_edit_distance=2, transfer_casing=True)
    #                 suggestion = suggestions[0].term
    #                 keyboard.set_field_text("suggestion", suggestion)
    #                 break
    #
    #             if len(key[0].name) <= 1:
    #                 character = key[0].name
    #             else:
    #                 character = SPECIAL_CHARACTERS.get(key[0].name)
    #
    #             text += character
    #
    #             highlights[key[0].name] = [3]
    #             keyboard.run(highlights, FEEDBACK_TIME,
    #                          start_marker=["start_feedback"],
    #                          stop_marker=["stop_feedback"])
    #             highlights[key[0].name] = [0]
    #             keyboard.set_field_text("text", text)
    #
    #             suggestions = sym_spell.lookup_compound(text, max_edit_distance=2, transfer_casing=True)
    #             suggestion = suggestions[0].term
    #             keyboard.set_field_text("suggestion", suggestion)
    #             break
    #
    #     if not key_pressed:
    #         logger.info('no key pressed')


    # Stop experiment


    keyboard.log(marker=["stop_experiment"])
    keyboard.log(marker=[f"Number of required trials: {i}"])
    keyboard.window.setMouseVisible(True)
    keyboard.set_field_text("text", "Experiment finished. Press button to close.")
    keyboard.set_field_text("suggestion", "")
    logger.info("Experiment finished. Press button to close.")
    event.waitKeys()
    keyboard.set_field_text("text", "Closing...")
    logger.info("Closing.")
    keyboard.run(highlights, 5.0)
    keyboard.set_field_text("text", "")
    keyboard.quit()
    logger.info("Experiment closed.")

    return 0

if __name__ == "__main__":
    import argparse

    keyboard = Keyboard(size=SCREEN_SIZE, width=SCREEN_WIDTH, distance=SCREEN_DISTANCE, screen=SCREEN,
                        window_color=SCREEN_COLOR, stream=STREAM)
    parser = argparse.ArgumentParser(description="Test keyboard.py")
    parser.add_argument("-n", "--ntrials", type=int, help="number of trials", default=5)
    parser.add_argument("-c", "--code", type=str, help="code set to use", default="onoff")
    args = parser.parse_args()
    test(n_trials=args.ntrials, code=args.code)