import cv2
import numpy as np
import os

# --- Setup ---
WIDTH, HEIGHT = 1200, 800  # Kept the expanded drawing area

canvas = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
drawing = False
start_pt = None
goal_pt = None
DATA_FILE = "map_data.npz"

# --- TERMINAL MENU ---
print("--- MAP MENU ---")
print("1. Draw a completely new map")

if os.path.exists(DATA_FILE):
    print("2. Modify existing map (Update Start/Goal or add obstacles)")
    choice = input("Select an option (1 or 2): ").strip()
    
    if choice == '2':
        try:
            data = np.load(DATA_FILE)
            canvas = data['canvas']
            start_pt = tuple(data['start'])
            goal_pt = tuple(data['goal'])
            print("Successfully loaded existing map.")
        except Exception as e:
            print(f"Error loading map: {e}. Starting fresh.")
            canvas = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
else:
    print(f"No existing '{DATA_FILE}' found. Starting fresh.")
    choice = '1'

# --- OPENCV EDITOR ---
def draw_callback(event, x, y, flags, param):
    global drawing, start_pt, goal_pt
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            start_pt = (y, x) # Store as (row, col)
        else:
            drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # Reverted back to the original 12px brush
        cv2.circle(canvas, (x, y), 12, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        goal_pt = (y, x)

cv2.namedWindow("1. Draw Shape", cv2.WINDOW_NORMAL)
cv2.resizeWindow("1. Draw Shape", WIDTH, HEIGHT)
cv2.setMouseCallback("1. Draw Shape", draw_callback)

print("\n--- EDITOR MODE ---")
print("Draw: Left Click | Start: Shift+Left Click | Goal: Right Click")
print("Press 'S' to Smooth, Save and Exit | 'C' to Clear")

while True:
    img_bin = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    
    # Reverted markers back to original size 6
    if start_pt: cv2.circle(img_bin, (start_pt[1], start_pt[0]), 6, (0, 255, 0), -1)
    if goal_pt: cv2.circle(img_bin, (goal_pt[1], goal_pt[0]), 6, (0, 0, 255), -1)
    
    cv2.imshow("1. Draw Shape", img_bin)
    key = cv2.waitKey(1)
    
    if key == ord('s'):
        if start_pt and goal_pt:
            # Reverted back to the original 11x11 blur kernel
            smoothed = cv2.GaussianBlur(canvas, (11, 11), 0)
            _, canvas = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
            
            # 2. Save the pristine, smoothed map
            np.savez(DATA_FILE, canvas=canvas, start=start_pt, goal=goal_pt)
            print(f"Map smoothed and saved to {DATA_FILE}")
            
            # 3. Redisplay the smoothed map so the user can verify the change
            final_display = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            cv2.circle(final_display, (start_pt[1], start_pt[0]), 6, (0, 255, 0), -1)
            cv2.circle(final_display, (goal_pt[1], goal_pt[0]), 6, (0, 0, 255), -1)
            
            cv2.imshow("1. Draw Shape", final_display)
            print("Previewing smoothed map. Press ANY KEY to exit...")
            cv2.waitKey(0) # Pauses the script here until you press a key
            break
        else:
            print("Error: Set both Start and Goal points before saving!")
            
    elif key == ord('c'):
        canvas[:] = 0 # Clears obstacles, but keeps your start/goal markers
        
    elif key == 27: # ESC
        break

cv2.destroyAllWindows()