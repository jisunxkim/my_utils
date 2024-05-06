from IPython.display import Audio

def sound_success():
    try:
        notify_success = Audio("applause2_x.wav", autoplay=True)
    except:
        pass
def sound_fail():
    try:
        notify_fail = Audio("hit_with_frying_pan_y.wav", autoplay=True)
    except:
        pass