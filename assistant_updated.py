
import pyautogui
import speech_recognition as sr
import pyttsx3

from sklearn.cluster import KMeans
import numpy as np

class UserBehaviorAnalyzer:
    def __init__(self):
        self.model = KMeans(n_clusters=3)

    def analyze_behavior(self, user_data):
        data = np.array(user_data)  # data from user behavior
        self.model.fit(data)
        return self.model.predict(data)

# Using the behavior analysis in the assistant
behavior_analyzer = UserBehaviorAnalyzer()
user_data = [[1, 2], [3, 4], [5, 6]]  # Example user data
user_behavior = behavior_analyzer.analyze_behavior(user_data)
self.speak(f"User behavior classified as: {user_behavior}")

import openai
import logging
from datetime import datetime
import random
import os
import time
from subprocess import Popen

class AdvancedPersonalAssistant:

    def __init__(self):
        self.engine = pyttsx3.init()  # For text-to-speech (TTS)
        self.recognizer = sr.Recognizer()  # For speech recognition (STT)

    
def interact_with_freelancing_platform(platform, action):
    if platform == "Upwork":
        if action == "check_messages":
            self.speak("Checking new messages on Upwork...")
            # Code to fetch messages from Upwork API
        elif action == "apply_for_jobs":
            self.speak("Applying for jobs based on your skills...")
            # Code to automatically apply for jobs
        elif action == "auto_reply":
            self.speak("Replying to the message...")
            # Code to send predefined reply to a message

    elif platform == "Fiverr":
        if action == "check_messages":
            self.speak("Checking your new messages on Fiverr...")
            # Code to fetch messages from Fiverr API
        elif action == "apply_for_gigs":
            self.speak("Applying for gigs based on your expertise...")
            # Code to automatically apply for gigs
        elif action == "auto_reply":
            self.speak("Sending a predefined reply to the message...")
            # Code to send a predefined reply to a message

def speak(self, text):
        # Speak the given text using pyttsx3 in a natural and smooth voice.
        self.engine.setProperty('rate', 150)  # Speed of speech (Normal)
        self.engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)
        self.engine.say(text)
        self.engine.runAndWait()

    
def interact_with_game(game_name, event_type):
    if game_name == "PUBG":
        if event_type == "kill":
            self.speak("Great job! You've made a kill!")
        elif event_type == "victory":
            self.speak("Congratulations on your victory! Well played!")
        elif event_type == "defeat":
            self.speak("Don't worry, try again next time!")
        else:
            self.speak("Keep playing, you're doing great!")

    elif game_name == "Free Fire":
        if event_type == "kill":
            self.speak("Nice kill, keep it up!")
        elif event_type == "victory":
            self.speak("Victory is yours! Fantastic work!")
        elif event_type == "defeat":
            self.speak("You’ll get them next time, don’t give up!")

def listen(self):
        # Listen for user input and convert it to text using Speech Recognition.
        with sr.Microphone() as source:
            print("Listening for commands...")
            self.speak("I'm listening, what can I do for you?")
            audio = self.recognizer.listen(source)
            try:
                command = self.recognizer.recognize_google(audio)
                print(f"User said: {command}")
                return command.lower()
            except sr.UnknownValueError:
                print("Sorry, I didn't understand that.")
                self.speak("Sorry, I didn't catch that. Could you repeat?")
                return "Sorry, I didn't understand that."
            except sr.RequestError:
                print("Could not request results.")
                self.speak("There was an error connecting. Please try again later.")
                return "Sorry, there was an error."

    def automate_task(self, task_name):
        # Automate tasks based on user command.
        if task_name == "open_browser":
            pyautogui.hotkey("ctrl", "t")  # Open a new tab in the browser
            self.speak("Opening a new browser tab")
            logging.info("Opening a new browser tab.")
        elif task_name == "take_screenshot":
            screenshot = self.capture_screen()
            self.speak("Screenshot taken successfully")
            logging.info(f"Screenshot saved at {screenshot}.")
        elif task_name == "close_browser":
            pyautogui.hotkey("ctrl", "w")  # Close the current tab
            self.speak("Closing the current browser tab")
            logging.info("Closing the current browser tab.")
        elif task_name == "open_photoshop":
            self.open_adobe_application("Photoshop")
        elif task_name == "open_illustrator":
            self.open_adobe_application("Illustrator")
        elif task_name == "open_premiere":
            self.open_adobe_application("Premiere Pro")
        elif task_name == "check_freelance_messages":
            self.check_freelance_platforms("messages")
        else:
            self.speak("I don't understand the task")

    def open_adobe_application(self, app_name):
        # Open Adobe application based on user command.
        app_paths = {
            "Photoshop": "C:\Program Files\Adobe\Adobe Photoshop.exe",
            "Illustrator": "C:\Program Files\Adobe\Adobe Illustrator.exe",
            "Premiere Pro": "C:\Program Files\Adobe\Adobe Premiere Pro.exe"
        }
        if app_name in app_paths:
            Popen(app_paths[app_name])  # Open the application
            self.speak(f"Opening {app_name}")
            logging.info(f"Opening {app_name} application.")
        else:
            self.speak(f"Sorry, I cannot open {app_name}.")
            logging.info(f"Failed to open {app_name}.")

    def check_freelance_platforms(self, task_type):
        # Check messages or new updates on freelance platforms.
        if task_type == "messages":
            self.speak("Checking your freelance messages on Upwork and Fiverr.")
            logging.info("Checking freelance messages.")
        else:
            self.speak("I'm not sure what you want me to check.")
            logging.info("Freelance task unknown.")

    def get_response_from_gpt(self, prompt):
        # Get a response from GPT-4 for a user query.
        openai.api_key = "your-openai-api-key"
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            temperature=0.7  # Control response creativity
        )
        return response.choices[0].text.strip()

    def handle_user_query(self, query):
        # Handle user queries intelligently.
        response = self.get_response_from_gpt(query)
        self.speak(response)
        logging.info(f"User query: {query} | Assistant response: {response}")
        return response

    def assist_with_tasks(self):
        # Main loop to assist user with tasks, listen for commands.
        while True:
            query = self.listen()
            if "stop" in query or "exit" in query:
                self.speak("Goodbye! Have a great day!")
                break
            elif "open browser" in query:
                self.automate_task("open_browser")
            elif "take screenshot" in query:
                self.automate_task("take_screenshot")
            elif "open photoshop" in query:
                self.automate_task("open_photoshop")
            elif "check freelance messages" in query:
                self.automate_task("check_freelance_messages")
            else:
                self.handle_user_query(query)
