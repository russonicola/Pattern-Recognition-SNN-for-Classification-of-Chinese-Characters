'''
Created on 15.12.2014

@author: Peter U. Diehl
'''
import os
import tkinter.filedialog
import time
from win32 import win32api, win32gui, win32print
from win32.lib import win32con
import preprocess
from win32.win32api import GetSystemMetrics
import tkinter as tk
from PIL import ImageGrab
from PIL import Image, ImageTk
import numpy as np
import os.path
import brian2 as b
import time
import pickle
import cv2 as cv
from brian2 import *


#------------------------------------------------------------------------------
# define the functions in network
#------------------------------------------------------------------------------
def get_labeled_data(picklename):
    data = pickle.load(open('%s.pickle' % picklename, 'rb'))
    return data


def recognize_number(assignments, spike_rates):
    summed_rates = [0] * 8
    num_assignments = [0] * 8
    for i in range(8):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]


def get_new_assignments(result_monitor, input_numbers):
    assignments = np.ones(n_e) * -1 # initialize them as not assigned
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e
    for j in range(8):
        num_inputs = len(np.where(input_nums == j)[0])
        if num_inputs > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_inputs
        for i in range(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j
    return assignments


def get_matrix_from_file(fileName, n_src, n_tgt):
    readout = np.load(fileName)
    print(readout.shape, fileName)
    value_arr = np.zeros((n_src, n_tgt))
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
    return value_arr


def normalize_weights():
    len_source = len(connections['XeAe'].source)
    len_target = len(connections['XeAe'].target)
    connection = np.zeros((len_source, len_target))
    connection[connections['XeAe'].i, connections['XeAe'].j] = connections['XeAe'].w
    temp_conn = np.copy(connection)
    colSums = np.sum(temp_conn, axis = 0)
    colFactors = 78./colSums
    for j in range(n_e):
        temp_conn[:,j] *= colFactors[j]
    connections['XeAe'].w = temp_conn[connections['XeAe'].i, connections['XeAe'].j]


def run_network():
    global input_intensity, previous_spike_count, t
    image = cv.imread('temp.png', cv.IMREAD_GRAYSCALE)
    if image is None:
        txt.insert(tk.END, 'No image input\n')
    else:
        image = preprocess.main(image, (50, 50))
        j = 0
        while j < 1:
            rate = image.reshape((n_input)) / 8. * input_intensity
            input_groups['Xe'].rates = rate * Hz
            net.run(single_example_time, report='text')
            current_spike_count = np.asarray(spike_counters['Ae'].count[:]) - previous_spike_count
            previous_spike_count = np.copy(spike_counters['Ae'].count[:])
            if np.sum(current_spike_count) < 5:
                input_intensity += 1
                input_groups['Xe'].rates = 0 * Hz
                net.run(resting_time)
            else:
                result_monitor = current_spike_count
                input_groups['Xe'].rates = 0 * Hz
                net.run(resting_time)
                input_intensity = start_input_intensity
                j += 1

        while t < 1:
            rate = image.reshape((n_input)) / 8. * input_intensity
            input_groups['Xe'].rates = rate * Hz
            net.run(single_example_time, report='text')
            current_spike_count = np.asarray(spike_counters['Ae'].count[:]) - previous_spike_count
            previous_spike_count = np.copy(spike_counters['Ae'].count[:])
            if np.sum(current_spike_count) < 5:
                input_intensity += 1
                input_groups['Xe'].rates = 0 * Hz
                net.run(resting_time)
            else:
                result_monitor = current_spike_count
                input_groups['Xe'].rates = 0 * Hz
                net.run(resting_time)
                input_intensity = start_input_intensity
                t += 1

        character_print = np.array(['后: backward', '前: forward', '上: up', '下: down', '左: left', '右: right', '入: entrance', '出: exit'])
        assignments = np.load("assignments.npy")

        test_results = recognize_number(assignments, result_monitor)
        txt.insert(tk.END, character_print[int(test_results[0])]+'\n')
#------------------------------------------------------------------------------
# define the functions in interface
#------------------------------------------------------------------------------

def get_real_resolution():
    """获取真实的分辨率"""
    hDC = win32gui.GetDC(0)
    # 横向分辨率
    w = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)
    # 纵向分辨率
    h = win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)
    return w, h


def get_screen_size():
    """获取缩放后的分辨率"""
    w = GetSystemMetrics(0)
    h = GetSystemMetrics(1)
    return w, h


real_resolution = get_real_resolution()
screen_size = get_screen_size()

# Windows 设置的屏幕缩放率
# ImageGrab 的参数是基于显示分辨率的坐标，而 tkinter 获取到的是基于缩放后的分辨率的坐标
screen_scale_rate = round(real_resolution[0] / screen_size[0], 2)


class Box:

    def __init__(self):
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None

    def isNone(self):
        return self.start_x is None or self.end_x is None

    def setStart(self, x, y):
        self.start_x = x
        self.start_y = y

    def setEnd(self, x, y):
        self.end_x = x
        self.end_y = y

    def box(self):
        lt_x = min(self.start_x, self.end_x)
        lt_y = min(self.start_y, self.end_y)
        rb_x = max(self.start_x, self.end_x)
        rb_y = max(self.start_y, self.end_y)
        return lt_x, lt_y, rb_x, rb_y

    def center(self):
        center_x = (self.start_x + self.end_x) / 2
        center_y = (self.start_y + self.end_y) / 2
        return center_x, center_y


class SelectionArea:

    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas
        self.area_box = Box()

    def empty(self):
        return self.area_box.isNone()

    def setStartPoint(self, x, y):
        self.canvas.delete('area', 'lt_txt', 'rb_txt')
        self.area_box.setStart(x, y)
        # 开始坐标文字
        self.canvas.create_text(
            x, y - 10, text=f'({x}, {y})', fill='red', tag='lt_txt')

    def updateEndPoint(self, x, y):
        self.area_box.setEnd(x, y)
        self.canvas.delete('area', 'rb_txt')
        box_area = self.area_box.box()
        # 选择区域
        self.canvas.create_rectangle(
            *box_area, fill='black', outline='red', width=2, tags="area")
        self.canvas.create_text(
            x, y + 10, text=f'({x}, {y})', fill='red', tag='rb_txt')


class ScreenShot():
    def __init__(self, scaling_factor=2):
        global img
        self.win = tk.Tk()
        # self.win.tk.call('tk', 'scaling', scaling_factor)
        self.width = self.win.winfo_screenwidth()
        self.height = self.win.winfo_screenheight()

        # 无边框，没有最小化最大化关闭这几个按钮，也无法拖动这个窗体，程序的窗体在Windows系统任务栏上也消失
        self.win.overrideredirect(True)
        self.win.attributes('-alpha', 0.25)

        self.is_selecting = False

        # 绑定按 Enter 确认, Esc 退出
        self.win.bind('<KeyPress-Escape>', self.exit)
        self.win.bind('<KeyPress-Return>', self.confirmScreenShot)
        self.win.bind('<Button-1>', self.selectStart)
        self.win.bind('<ButtonRelease-1>', self.selectDone)
        self.win.bind('<Motion>', self.changeSelectionArea)

        self.canvas = tk.Canvas(self.win, width=self.width,
                                height=self.height)
        self.canvas.pack()
        self.area = SelectionArea(self.canvas)
        self.win.mainloop()
        #os.remove('temp.png')

    def exit(self, event):
        self.win.destroy()

    def clear(self):
        self.canvas.delete('area', 'lt_txt', 'rb_txt')
        self.win.attributes('-alpha', 0)

    def captureImage(self):
        global img
        if self.area.empty():
            return None
        else:
            box_area = [x * screen_scale_rate for x in self.area.area_box.box()]
            self.clear()
            print(f'Grab: {box_area}')
            img = ImageGrab.grab(box_area)
            return img

    def confirmScreenShot(self, event):
        global img_gif
        img = self.captureImage()
        if img is not None:
            img = img.resize((400,400))
            img.save('temp.png')
        self.win.destroy()
        # winNew = tk.Toplevel(root)
        # winNew.geometry('500x500')
        # winNew.title('截屏')
        # img_gif = ImageTk.PhotoImage(Image.open('temp.gif'))
        # label_img = tk.Label(winNew, image=img_gif)
        # label_img.pack()
        # txt.insert(tk.END, "上\n")
        root.state('normal')

    def selectStart(self, event):
        self.is_selecting = True
        self.area.setStartPoint(event.x, event.y)
        # print('Select', event)

    def changeSelectionArea(self, event):
        if self.is_selecting:
            self.area.updateEndPoint(event.x, event.y)
            # print(event)

    def selectDone(self, event):
        # self.area.updateEndPoint(event.x, event.y)
        self.is_selecting = False


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.place()
        self.createWidget()

    def createWidget(self):
        global photo
        photo = None
        self.label03 = tk.Label(self, image=photo)
        self.label03.grid(column=0, row=1)

        self.btn01 = tk.Button(self, text='Open file', command=self.getfile, bg='white', anchor='s')
        self.btn01.grid(column=0, row=0)


    def getfile(self):
        file_path = tk.filedialog.askopenfilename(title='Choose file', filetypes=[(('JPG', '*.jpg')), ('All Files', '*')])
        if file_path != '':
            img = Image.open(file_path)
            img = img.resize((400, 400))

            global photo
            photo = ImageTk.PhotoImage(img)
            self.label03.configure(image=photo)
            self.label03.image = photo
            img.save('temp.png')
        else:
            print('file_path is None')


def screenShot():
    root.state('icon')
    time.sleep(0.2)
    ScreenShot()


def input_image():
    root.state('icon')
    winNew = tk.Toplevel(root)
    winNew.geometry('500x500')
    winNew.title('Input your image')
    Application(master=winNew)


# start place
np.random.seed(0)
data_path = './'
n_input = 2500
n_e = 400
n_i = n_e
single_example_time = 0.35 * b.second #
resting_time = 0.15 * b.second
runtime = single_example_time + resting_time

v_rest_e = -65. * b.mV
v_rest_i = -60. * b.mV
v_reset_e = -65. * b.mV
v_reset_i = 'v=-45.*mV'
v_thresh_i = 'v>-40.*mV'
refrac_e = 5. * b.ms
refrac_i = 2. * b.ms

input_intensity = 2.
start_input_intensity = input_intensity

tc_pre_ee = 20*b.ms
tc_post_1_ee = 20*b.ms
tc_post_2_ee = 40*b.ms
nu_ee_pre =  0.0001      # learning rate
nu_ee_post = 0.01       # learning rate
wmax_ee = 1.0

scr_e = 'v = v_reset_e; timer = 0*ms'
offset = 20.0*b.mV
v_thresh_e = '(v>(theta - offset + -52.*mV)) and (timer>refrac_e)'
neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-100.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''
neuron_eqs_e += '\n  theta      :volt'
neuron_eqs_e += '\n  dtimer/dt = 0.1  : second'
neuron_eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-85.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''

eqs_stdp_ee = '''
                post2before                            : 1
                dpre/dt   =   -pre/(tc_pre_ee)         : 1 (event-driven)
                dpost1/dt  = -post1/(tc_post_1_ee)     : 1 (event-driven)
                dpost2/dt  = -post2/(tc_post_2_ee)     : 1 (event-driven)
            '''
eqs_stdp_pre_ee = 'pre = 1.; w = clip(w - nu_ee_pre * post1, 0, wmax_ee)'
eqs_stdp_post_ee = 'post2before = post2; w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee); post1 = 1.; post2 = 1.'

neuron_groups = {}
input_groups = {}
connections = {}
spike_counters = {}

neuron_groups['Ae'] = b.NeuronGroup(n_e, neuron_eqs_e, threshold= v_thresh_e, refractory= refrac_e, reset= scr_e, method='euler')
neuron_groups['Ai'] = b.NeuronGroup(n_i, neuron_eqs_i, threshold= v_thresh_i, refractory= refrac_i, reset= v_reset_i, method='euler')


#------------------------------------------------------------------------------
# create network population and recurrent connections
#------------------------------------------------------------------------------
print('create neuron group A')

neuron_groups['Ae'].v = v_rest_e - 40. * b.mV
neuron_groups['Ai'].v = v_rest_i - 40. * b.mV
neuron_groups['Ae'].theta = np.load(data_path + 'weights_c/theta_A_2000.npy') * b.volt
print('create recurrent connections')
weightMatrix = get_matrix_from_file(data_path + 'random_c/AeAi.npy', n_e, n_i)
connections['AeAi'] = b.Synapses(neuron_groups['Ae'], neuron_groups['Ai'], model='w : 1', on_pre='ge_post += w')
connections['AeAi'].connect(True) # all-to-all connection
connections['AeAi'].w = weightMatrix[connections['AeAi'].i, connections['AeAi'].j]

weightMatrix = get_matrix_from_file(data_path + 'random_c/AiAe.npy', n_i, n_e)
connections['AiAe'] = b.Synapses(neuron_groups['Ai'], neuron_groups['Ae'], model='w : 1', on_pre='gi_post += w')
connections['AiAe'].connect(True) # all-to-all connection
connections['AiAe'].w = weightMatrix[connections['AiAe'].i, connections['AiAe'].j]

print('create monitors for Ae')
spike_counters['Ae'] = b.SpikeMonitor(neuron_groups['Ae'])

#------------------------------------------------------------------------------
# create input population and connections from input populations
#------------------------------------------------------------------------------
input_groups['Xe'] = b.PoissonGroup(n_input, 0*Hz)

print('create connections between X and A')
weightMatrix = get_matrix_from_file(data_path + 'weights_c/XeAe_2000.npy', n_input, n_e)

model = 'w : 1'
pre = 'ge_post += w'
post = ''

connections['XeAe'] = b.Synapses(input_groups['Xe'], neuron_groups['Ae'],
                                            model=model, on_pre=pre, on_post=post)
minDelay = 0*b.ms
maxDelay = 10*b.ms
deltaDelay = maxDelay - minDelay

connections['XeAe'].connect(True) # all-to-all connection
connections['XeAe'].delay = 'minDelay + rand() * deltaDelay'
connections['XeAe'].w = weightMatrix[connections['XeAe'].i, connections['XeAe'].j]


#------------------------------------------------------------------------------
# run the simulation and set inputs
#------------------------------------------------------------------------------

net = Network()
for obj_list in [neuron_groups, input_groups, connections, spike_counters]:
    for key in obj_list:
        net.add(obj_list[key])

previous_spike_count = np.zeros(n_e)
input_groups['Xe'].rates = 0 * Hz
net.run(0*second)


#------------------------------------------------------------------------------
# interface implement
#------------------------------------------------------------------------------
root = tk.Tk()
root.title('User Interface')
# 指定窗口的大小
root.geometry('500x400')
# 不允许改变窗口大小
root.resizable(True, True)
image = cv.imread('temp.png', cv.IMREAD_GRAYSCALE)
t = 0
# ================== 布置截屏按钮 ====================================
button_screenShot = tk.Button(root, text='Screenshot', command=screenShot)
button_screenShot.place(relx=0.25, rely=0.05, relwidth=0.5, relheight=0.18)
button_input = tk.Button(root, text='Input image', command=input_image)
button_input.place(relx=0.25, rely=0.28, relwidth=0.5, relheight=0.18)
button_recognize = tk.Button(root, text='Recognition result', command=run_network)
button_recognize.place(relx=0.35, rely=0.5, relwidth=0.3, relheight=0.10)
txt = tk.Text(root)
txt.place(rely=0.6, relheight=0.4)
# ================== 完 =============================================

try:
    root.mainloop()
except:
    root.destroy()
