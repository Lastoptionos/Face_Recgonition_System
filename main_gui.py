import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk  # PIL is used to handle images in tkinter
import subprocess
import os

# Define functions for each operation
def capture_faces():
    messagebox.showinfo("Capture Faces", "Starting face capture...")
    subprocess.run(["python", "capture_faces.py"])  # Assuming capture_faces.py is your script for face capture

def recognize_faces():
    messagebox.showinfo("Face Recognition", "Starting real-time face recognition...")
    subprocess.run(["python", "face_recognition.py"])  # Assuming recognize_faces.py is your script for face recognition

def show_users():
    file_path = 'dataset/labels.txt'
    
    # Check if the file exists
    if not os.path.exists(file_path):
        messagebox.showinfo("Show Users", "No users found. The labels file does not exist.")
        return

    with open(file_path, 'r') as file:
        labels = [label.strip() for label in file.readlines() if label.strip()]  # Remove empty lines and strip whitespace

    if not labels:
        messagebox.showinfo("Show Users", "No users found in the labels file.")
        return

    # Create a popup window to display users
    user_window = tk.Toplevel(root)
    user_window.title("Registered Users")
    user_window.geometry("300x300")
    user_window.configure(bg='#2C3E50')

    tk.Label(user_window, text="Registered Users", bg='#2C3E50', fg='white', font=("Helvetica", 14, "bold")).pack(pady=10)

    user_listbox = tk.Listbox(user_window, bg='#34495E', fg='white', font=("Helvetica", 12), selectbackground='#1ABC9C')
    user_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    for index, label in enumerate(labels, start=0):
        user_listbox.insert(tk.END, f"{index}. {label}")

    ttk.Button(user_window, text="Close", command=user_window.destroy).pack(pady=10)

# Create the main window
root = tk.Tk()
root.title("Face Recognition System")

# Set window size and background color
root.geometry("500x400")
root.configure(bg='#2C3E50')

# Add a title label
title_label = tk.Label(root, text="Face Recognition System", bg='#2C3E50', fg='white', font=("Helvetica", 16, "bold"))
title_label.pack(pady=20)

# Load images for the buttons
capture_img = Image.open("icons/capture.png")
capture_img = capture_img.resize((50, 50), Image.LANCZOS)  # Use LANCZOS instead of ANTIALIAS
capture_icon = ImageTk.PhotoImage(capture_img)

recognize_img = Image.open("icons/recognize.png")
recognize_img = recognize_img.resize((50, 50), Image.LANCZOS)
recognize_icon = ImageTk.PhotoImage(recognize_img)

show_users_img = Image.open("icons/list.png")  # Add an icon for "Show Users"
show_users_img = show_users_img.resize((50, 50), Image.LANCZOS)
show_users_icon = ImageTk.PhotoImage(show_users_img)

# Add buttons for each operation, horizontally aligned
button_frame = tk.Frame(root, bg='#2C3E50')
button_frame.pack(pady=20)

capture_button = ttk.Button(button_frame, text="Capture Faces", command=capture_faces, image=capture_icon, compound="top", style="TButton")
capture_button.grid(row=0, column=0, padx=20)

recognize_button = ttk.Button(button_frame, text="Face Recognition", command=recognize_faces, image=recognize_icon, compound="top", style="TButton")
recognize_button.grid(row=0, column=1, padx=20)

show_users_button = ttk.Button(button_frame, text="Show Users", command=show_users, image=show_users_icon, compound="top", style="TButton")
show_users_button.grid(row=0, column=2, padx=20)

# Add a status bar
status_var = tk.StringVar()
status_var.set("Ready")
status_bar = tk.Label(root, textvariable=status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, bg='#34495E', fg='white')
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# Start the GUI event loop
root.mainloop()
