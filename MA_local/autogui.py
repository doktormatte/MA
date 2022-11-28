import pyautogui
import os
import time
import random
import numpy as np



def pause(n):
    for i in range(n):
        time.sleep(random.choice([random.uniform(1.0, 1.5),np.random.exponential(1),np.random.exponential(1.5)])) 
        
i = 0
try:
    pyautogui.hotkey('alt', 'tab')
    # while i < 20:
    while True:
        pause(2)        
        pos_x = random.randint(300, 3000)
        pos_y = random.randint(1000,2000)       
        pause(1)
        pyautogui.moveTo(pos_x, pos_y)
        pause(1)
        scroll_len = random.randint(-10, 10)
        # if random.randint(0, 3) == 1:
        #     pause(1)
        #     pyautogui.click() 
        pyautogui.scroll(scroll_len)
        pause(1)
        i += 1        
        
except KeyboardInterrupt:
    pass



# try:
#     pyautogui.hotkey('alt', 'tab')
#     pos_x = 2000
#     pos_y = 1000
#     pause(1)
#     pyautogui.moveTo(pos_x, pos_y)
#     pyautogui.click() 
#     pause(1)
#     pyautogui.hotkey('ctrl', 'b')
#     pause(1)
    
    
#     # while i < 20:
#     while True:
#         pause(2)        
#         pos_x = random.randint(300, 3000)
#         pos_y = random.randint(1000,2000)       
#         pause(1)
#         pyautogui.moveTo(pos_x, pos_y)
#         pause(1)
#         scroll_len = random.randint(-10, 10)
#         pyautogui.scroll(scroll_len)
#         pause(1)
#         i += 1        
        
except KeyboardInterrupt:
    pass