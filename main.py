import pyautogui
import keyboard
import cv2
import numpy as np

# Flag to check if the game has started
game_started = False

# Variables to store left and right extremes of the board
left_extreme = None
right_extreme = None

# Infinite loop to continuously capture and display the screen
while True:
    try:
        # Look for the board on the screen and store bounding values
        if not game_started:
            board_location = pyautogui.locateOnScreen('board.png', confidence=0.5)
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
            else:
                print("game not found")

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
