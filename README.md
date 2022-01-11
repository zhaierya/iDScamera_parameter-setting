# iDScamera_parameter-setting
setting the exposure time and trigger mode(camera as input)

uEye_class is the file that can open the iDS camera and run it in live.

i am aiming at controlling the exposure time and trigger mode

exposure time can not reach out the range of the camera, so it should be not more than 70
trigger mode i habe set 2 module, one is live mode, it means no trigger
the other is external falling edge trigger(you can change it in the code), 
and i donnt have a external hardware trigger, in def triggered_video(self): i have set a dummy trigger signal using ueye.is_ForceTrigger
when you haber a real hardaware trigger, you should use ueye.is_FreezeVideo to capture the image

the numer after the filename uEye_class is the data i update the file
the format of the data is DDMM (day month)
i think the newst data is the beste code, should get less problem

