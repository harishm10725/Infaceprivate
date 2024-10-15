import streamlit as st
import os
from newimage import create_new_faces
from recognize import recognize_face
from triplet import create_triplet_path,path_extracter
from resnetembedder import siamese_network

directory = r"E:\DATA SCIENCE\programs\InFace\images"

def file_exists(directory,filename ):
    path = os.path.join(directory,filename)
    return os.path.exists(path)
# Add a header to the Streamlit app
st.title("Student Registration System")

# Ask the user if they are a new student using a radio button
is_new_student = st.radio("Are you a new student?", ("Yes", "No"))

if is_new_student == "Yes":
    # Text input for the register ID
    register_id = st.text_input("Enter your register ID:")

    if st.button("Register"):  # Add a button to trigger the function
        if register_id:
            if not file_exists(directory,register_id):
               create_new_faces(register_id)  # Call the function to create new faces
               st.success(f"Faces created for register ID: {register_id}")
            else:
               st.error("Register ID already existsc.")
        else:
            st.error("Please enter a valid register ID.")
else:

    anchor = recognize_face()
    labels = os.listdir(directory)
    triplet_paths = create_triplet_path(anchor,labels,directory)
    triplets = path_extracter(triplet_paths)
    outcome = siamese_network(triplets)
    if(outcome):
        st.success(f"Face recognized and result is True")
    else:
        st.error("Face does not recognize and result is False")

