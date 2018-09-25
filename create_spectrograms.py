import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import PIL.Image as Image
import os
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default='/movie/audio/topcoder/topcoder_train',help="")
parser.add_argument("--save_img_dir", type=str, default="/movie/audio/topcoder/topcoder_train_png",help="")
parser.add_argument("--audio_type", type=str, default="mp3",help="audio type")

opt = parser.parse_args()
print opt


""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)
    # cols for windowing
    cols = int(np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1)
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win
    
    return np.fft.rfft(frames)    
    
""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20., alpha=1.0, f0=0.9, fmax=1):
    spec = spec[:, 0:256]
    timebins, freqbins = np.shape(spec)
    scale = np.linspace(0, 1, freqbins) #** factor
    
    # http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=650310&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel4%2F89%2F14168%2F00650310
    scale = np.array(map(lambda x: x * alpha if x <= f0 else (fmax-alpha*f0)/(fmax-f0)*(x-f0)+alpha*f0, scale))
    scale *= (freqbins-1)/max(scale)

    newspec = np.complex128(np.zeros([timebins, freqbins]))
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = [0.0 for i in range(freqbins)]
    totw = [0.0 for i in range(freqbins)]
    for i in range(0, freqbins):
        if (i < 1 or i + 1 >= freqbins):
            newspec[:, i] += spec[:, i]
            freqs[i] += allfreqs[i]
            totw[i] += 1.0
            continue
        else:
            # scale[15] = 17.2
            w_up = scale[i] - np.floor(scale[i])
            w_down = 1 - w_up
            j = int(np.floor(scale[i]))
           
            newspec[:, j] += w_down * spec[:, i]
            freqs[j] += w_down * allfreqs[i]
            totw[j] += w_down
            
            newspec[:, j + 1] += w_up * spec[:, i]
            freqs[j + 1] += w_up * allfreqs[i]
            totw[j + 1] += w_up
    
    for i in range(len(freqs)):
        if (totw[i] > 1e-6):
            freqs[i] /= totw[i]
    
    return newspec, freqs

""" plot spectrogram"""
def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="gray", channel=0, name='tmp.png', alpha=1, offset=0):
    samplerate, samples = wav.read(audiopath)
    samples = samples if len(samples.shape) <=1 else samples[:, channel]
    s = stft(samples, binsize) # 431 * 513

    #sshow : 431 * 256,
    sshow, freq = logscale_spec(s, factor=1, sr=samplerate, alpha=alpha)
    sshow = sshow[2:, :]
    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel
    timebins, freqbins = np.shape(ims)

    ims = np.transpose(ims)
    # ims = ims[0:256, offset:offset+768] # 0-11khz, ~9s interval
    ims = ims[0:256, :] # 0-11khz, ~10s interval
    #print "ims.shape", ims.shape

    image = Image.fromarray(ims)
    image = image.convert('L')
    image.save(name)

def create_spec(input_dir,save_img_dir,audio_type):

        # file = open(input_dir, 'r')
        for iter, line in enumerate(glob.glob(os.path.join(input_dir,"*.%s"%audio_type))):#enumerate(file.readlines()[1:]): # first line of traininData.csv is header (only for trainingData.csv)
            # filepath = line.split(',')[0]
            filename = line.split("/")[-1].split(".")[0]
            # file_split.pop(-1)
            # filename = "_".join(file_split)
            if audio_type == "mp3":
                wavfile = os.path.join(input_dir,filename + '.wav')
                #os.system('ffmpeg -i ' + line +" -ar 8000 tmp.wav" )
                #os.system("ffmpeg -i tmp.wav -ar 8000 tmp.mp3" )
                #os.system("ffmpeg -i tmp.mp3 -ar 8000 -ac 1 "+wavfile )
                os.system('ffmpeg -i ' + line +" " + wavfile)

                # we create only one spectrogram for each speach sample
                # we don't do vocal tract length perturbation (alpha=1.0)
                # also we don't crop 9s part from the speech
                plotstft(wavfile, channel=0, name=os.path.join(save_img_dir,filename+'.png'), alpha=1.0)
                os.remove(wavfile)
                #os.remove("tmp.mp3")
                #os.remove("tmp.wav")
            elif audio_type == "wav":
                plotstft(line, channel=0, name=os.path.join(save_img_dir, filename + '_1.png'), alpha=1.0)
            print "processed %d files" % (iter + 1)


if __name__ == '__main__':
    create_spec(input_dir = opt.input_dir,save_img_dir = opt.save_img_dir,audio_type = opt.audio_type)
