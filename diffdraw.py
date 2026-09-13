import cv2
import numpy as np
import os

# --- Setup ---
WIDTH, HEIGHT = 1200, 800 # Increased canvas size
canvas = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
drawing = False
start_pt = None
goal_pt = None
DATA_FILE = "map_data.npz"

# New: Brush Selection
brush_color = 255 # Default to Rigid Wall
brush_name = "RIGID WALL (Far Clearance)"

print("--- MAP MENU ---")
print("1. Draw a completely new map")

if os.path.exists(DATA_FILE):
    print("2. Modify existing map")
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
else:
    print(f"No existing '{DATA_FILE}' found. Starting fresh.")

# --- OPENCV EDITOR ---
def draw_callback(event, x, y, flags, param):
    global drawing, start_pt, goal_pt, brush_color
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            start_pt = (y, x) 
        else:
            drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # Draw with the currently selected obstacle type
        cv2.circle(canvas, (x, y), 12, brush_color, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        goal_pt = (y, x)

# Added WINDOW_NORMAL so you can resize the window on your desktop
cv2.namedWindow("1. Draw Shape", cv2.WINDOW_NORMAL)
cv2.resizeWindow("1. Draw Shape", WIDTH, HEIGHT)
cv2.setMouseCallback("1. Draw Shape", draw_callback)

print("\n--- EDITOR MODE ---")
print("Draw: Left Click | Start: Shift+Left Click | Goal: Right Click")
print("Press '1' to use RIGID WALL Brush (White)")
print("Press '2' to use TIGHT CLEARANCE Brush (Gray)")
print("Press 'S' to Smooth, Save and Exit | 'C' to Clear")

while True:
    # Convert semantic 1-channel map to BGR for display
    img_disp = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    
    # Overlay Start/Goal
    if start_pt: cv2.circle(img_disp, (start_pt[1], start_pt[0]), 6, (0, 255, 0), -1)
    if goal_pt: cv2.circle(img_disp, (goal_pt[1], goal_pt[0]), 6, (0, 0, 255), -1)
    
    # Display Current Brush Status (Anchored to top-left, looks fine on large window)
    cv2.putText(img_disp, f"Brush: {brush_name}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    cv2.imshow("1. Draw Shape", img_disp)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('1'):
        brush_color = 255
        brush_name = "RIGID WALL (Far Clearance)"
    elif key == ord('2'):
        brush_color = 128
        brush_name = "TIGHT CLEARANCE (Close)"
    elif key == ord('s'):
        if start_pt and goal_pt:
            # Solidify shape while preserving the distinct 128/255 values
            smoothed = cv2.GaussianBlur(canvas, (11, 11), 0)
            
            # Snap values back to semantic categories after blur
            canvas[smoothed >= 190] = 255 # Restore Rigid
            canvas[(smoothed >= 60) & (smoothed < 190)] = 128 # Restore Tight
            canvas[smoothed < 60] = 0 # Free space
            
            np.savez(DATA_FILE, canvas=canvas, start=start_pt, goal=goal_pt)
            print(f"Map saved to {DATA_FILE}")
            break
        else:
            print("Error: Set both Start and Goal points!")
    elif key == ord('c'):
        canvas[:] = 0 
    elif key == 27: # ESC
        break

cv2.destroyAllWindows()