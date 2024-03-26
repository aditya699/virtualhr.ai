# Import library
import speech_recognition as sr
import pyttsx3

# Initialize recognizer class (for recognizing the speech)
r = sr.Recognizer()

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Reading Microphone as source
# Listening to the speech and storing it in audio_text variable
with sr.Microphone() as source:
    print("Please Ask Your Doubt")
    engine.say("Please Ask Your Doubt")
    engine.runAndWait()
    audio_text = r.listen(source)
    print("Time over, thanks")
    engine.say("Time over, thanks")
    engine.runAndWait()

# Recognize speech and convert it to text
try:
    # Using Google Speech Recognition
    recognized_text = r.recognize_google(audio_text)
    print("Text: " + recognized_text)
    engine.say("You said: " + recognized_text)
    engine.runAndWait()
except sr.UnknownValueError:
    print("Sorry, I did not understand that")
    engine.say("Sorry, I did not understand that")
    engine.runAndWait()
except sr.RequestError:
    print("Sorry, could not request results. Please check your internet connection")
    engine.say("Sorry, could not request results. Please check your internet connection")
    engine.runAndWait()
