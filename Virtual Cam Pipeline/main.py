import tkinter as tk
from tkinter import ttk
from multiprocessing import Process, Queue
from settings import edit_settings, clear_logs
from process_emotion import visualize_logs
import cv2
import numpy as np
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Global frame queue for inter-process communication
frame_queue = Queue()

def main():
    # Setup the main application window
    root = tk.Tk()
    root.title("Emoji Cam")
    root.geometry("250x290")
    root.configure(bg="white")
    style = ttk.Style(root)
    style.theme_use('clam')

    button_width = 20

    # Create buttons for the UI
    tk.Button(root,
              text="Enable Virtual Camera",
              command=start_multiprocess_display,
              height=2, width=button_width,
              bg="white", fg="black").pack(pady=8)

    tk.Button(root, text="Settings",
              command=lambda: edit_settings(root),
              height=2, width=button_width,
              bg="white", fg="black").pack(pady=8)

    tk.Button(root,
              text="Visualize Logs",
              command=visualize_logs,
              height=2, width=button_width,
              bg="white", fg="black").pack(pady=8)

    tk.Button(root,
              text="Clear Logs",
              command=clear_logs,
              height=2, width=button_width,
              bg="white", fg="black").pack(pady=8)

    tk.Button(root,
              text="Exit",
              command=root.destroy,
              height=2, width=button_width,
              bg="white", fg="black").pack(pady=8)

    root.mainloop()

def start_multiprocess_display():
    # Import the emotion processing loop
    from fer_pipeline import run_fer_loop

    # Start FER process
    logging.info("Starting FER process...")
    p = Process(target=run_fer_loop, args=(frame_queue,))
    p.daemon = True
    p.start()

    # Start the display loop in main process
    display_loop(frame_queue)

def display_loop(queue):
    try:
        while True:
            if not queue.empty():
                frame = queue.get()
                if isinstance(frame, np.ndarray):
                    cv2.imshow("Emoji Cam", frame)
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC to quit
                        break
    except Exception as e:
        logging.error(f"Display loop error: {e}")
    finally:
        cv2.destroyAllWindows()
        logging.info("Display window closed.")

if __name__ == "__main__":
    main()
