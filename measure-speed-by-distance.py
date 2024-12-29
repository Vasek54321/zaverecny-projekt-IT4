import cv2
import time
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

def calculate_speed(distance_meters, time_seconds):
    """
    Calculate speed in km/h.
    :param distance_meters: Distance in meters between the two points
    :param time_seconds: Time in seconds for the car to travel between points
    :return: Speed in km/h
    """
    speed_mps = distance_meters / time_seconds
    speed_kmph = speed_mps * 3.6
    return speed_kmph

def open_video():
    video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
    if not video_path:
        return

    distance_meters = simpledialog.askfloat("Input Distance", "Enter the real-world distance (in meters) between the two points:")
    if distance_meters is None:
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open video.")
        return

    times = []
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Point marked at: ({x}, {y})")
            if len(points) == 2:
                cv2.circle(frame, points[0], 5, (0, 255, 0), -1)
                cv2.circle(frame, points[1], 5, (0, 0, 255), -1)
                cv2.imshow("Video", frame)
                messagebox.showinfo("Points Marked", "Two points have been marked. Video will now continue.")

    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", click_event)

    # Show the first frame and wait for the user to mark points
    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Cannot read the first frame of the video.")
        cap.release()
        cv2.destroyAllWindows()
        return

    cv2.imshow("Video", frame)
    messagebox.showinfo("Mark Points", "Click on two points to mark the start and end points.")
    cv2.waitKey(0)  # Wait until points are marked

    if len(points) < 2:
        messagebox.showerror("Error", "Less than two points marked. Exiting.")
        cap.release()
        cv2.destroyAllWindows()
        return

    # Continue with video playback and measure time
    def measure_time(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            times.append(time.time())
            print(f"Time marked at: {times[-1]:.2f} seconds")
            if len(times) == 2:
                elapsed_time = times[1] - times[0]
                speed_kmph = calculate_speed(distance_meters, elapsed_time)
                messagebox.showinfo("Speed Calculation", f"Time between points: {elapsed_time:.2f} seconds\nCalculated speed: {speed_kmph:.2f} km/h")
                cap.release()
                cv2.destroyAllWindows()

    cv2.setMouseCallback("Video", measure_time)

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showinfo("End of Video", "End of video or cannot fetch frame.")
            break

        cv2.imshow("Video", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    root = tk.Tk()
    root.title("Car Speed Measurement")

    btn_open_video = tk.Button(root, text="Open Video", command=open_video)
    btn_open_video.pack(pady=20)

    btn_exit = tk.Button(root, text="Exit", command=root.quit)
    btn_exit.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()