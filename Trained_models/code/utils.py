# coding=utf-8

def load_dict(fasterRCNN, state_dict):
    try:
        fasterRCNN.load_state_dict(state_dict)
    except:
        ori = state_dict.keys()
        det = fasterRCNN.state_dict().keys()
        j=0
        for i in range(len(ori)):
            j+=1
            state_dict[det[i]] = state_dict.pop(ori[i])
        fasterRCNN.load_state_dict(state_dict)
        del ori
        del det