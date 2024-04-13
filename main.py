import pyautogui
import keyboard
import cv2
import numpy as np
from PIL import Image
import win32api, win32con

# Flag to check if the game has started
game_started = False

# Variables to store left and right extremes of the board
left_extreme = None
right_extreme = None


def click(x, y):
    win32api.SetCursorPos((int(x), int(y)))
    # win32api.mouse_event(win32con.)


# Infinite loop to continuously capture and display the screen
while True:
    try:
        # Look for the board on the screen and store bounding values
        if not game_started:
            for scale in range(5, 21):
                # Calculate the scale factor
                scale_factor = scale / 10.0

                # Open the image file
                img = Image.open('board.png')

                # Resize the image
                width, height = img.size
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                resized_img = img.resize((new_width, new_height))

                # Save the resized image temporarily
                resized_img.save('resized_board.png')
                region = (0, 0, pyautogui.size()[0], pyautogui.size()[1])  # Full screen region
                board_location = pyautogui.locateOnScreen('resized_board.png', region=region, confidence=0.5)
                if board_location is not None:
                    left, top, width, height = board_location

                    # Store left and right extremes
                    left_extreme = left
                    right_extreme = left + width

                    print("Board found at:", board_location)
                    print("Left extreme:", left_extreme)
                    print("Right extreme:", right_extreme)

                    # Simulate a keyboard press (space key)

                    keyboard.press_and_release('space')

                    # Set game_started to True to exit the loop
                    game_started = True
                    break
                else:
                    print("game not found")
                resized_img.close()

        # Capture screenshot of the board
        else:
            screenshot = pyautogui.screenshot(region=(left_extreme, top, right_extreme - left_extreme, height))
            screen_np = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

            # Display the board
            cv2.imshow('Frame', screen_np)

    except pyautogui.ImageNotFoundException as e:
        print("An error occurred:", e)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close OpenCV windows
cv2.destroyAllWindows()
