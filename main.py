import pyautogui
from pyautogui import*
import cv2
#import win32api
#import win32con

'''
def click(x,y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)
'''

game_started= False
left_extreme= None
right_extreme= None

while True:
    try:
        #look for the board on the screen and store bounding values
        if not game_started:
            board_location = pyautogui.locateOnScreen('board.png',confidence=0.8)

            if board_location:
                left_extreme = board_location.left
                right_extreme = board_location.left + board_location.width
                print("Board found at: ",board_location)
                print("Left extreme: ",left_extreme)
                print("Right extreme: ",right_extreme)
                break

    except ImageNotFoundException as e:
        print("Board not found: ",e)
