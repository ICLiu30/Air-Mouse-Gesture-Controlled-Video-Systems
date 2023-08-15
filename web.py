import streamlit as st
from pathlib import Path

dir = Path(__file__).resolve().parent 

mouse = dir / 'GIF' / 'Mouse.gif'
video = dir / 'GIF' / 'Video.gif'

st.write("## Air Mouse & Gesture-Controlled Video System")
st.write("It's a perfect Friday evening. You're settled at your computer desk, a mouth-watering sandwich on one side and a cold drink on the other, ready to binge-watch your favorite show. All's well until... you need to pause, rewind, or adjust the volume.")
st.write("You face a dilemma:")
st.write("1.Risk smudging your keyboard and mouse with mayo and crumbs?")
st.write("2.Set aside your delightful sandwich, making the heartbreaking choice of cleanliness over continuous entertainment?")
st.write("**The Solution:**")
st.write("Enter the **Air Mouse & Gesture-Controlled Video System**. With a mere wave of your (potentially messy) hand, you can take control without ever touching your hardware. Play, pause, skip, or adjust - all with intuitive gestures.")

st.write("### Air Mouse")
st.write("For the Air Mouse system, I harnessed MediaPipe, OpenCV, and PyAutoGUI to detect finger movements, enabling actions like scrolling, clicking, and cursor movement. Additionally, I integrated speech recognition for hands-free text input.")

st.image(mouse)
st.write("### Gesture-Controlled Video")
st.write("The Gesture-Controlled Video system is rooted in a Random Forest model. This model deciphers hand gestures to execute video commands, such as fast-forwarding, rewinding, pausing, or adjusting volume, making video navigation seamless and intuitive.")

st.image(video)

st.write("### Acknowledgements & Further Exploration")
st.markdown("I'd like to extend my heartfelt thanks to [Murtaza's Workshop](https://www.youtube.com/watch?v=8gPONnGIPgw&t=321) for being the foundational guide during the construction of these systems. For those intrigued by the code and training aspects of this project, I invite you to dive deeper into my [GitHub](https://github.com/ICLiu30/Air-Mouse-Gesture-Controlled-Video-Systems).")
