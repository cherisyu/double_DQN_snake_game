### coding: utf-8 ###

import tensorflow as tf
import cv2
import sys
sys.path.append("snake_game/")

import t2
import random

import numpy as np
from collections import deque
import copy

GAME = 'snake' # the name of the game being played for log files
ACTIONS = 4 # number of valid actions
EXPLORE = 2000000. # frames over which to anneal epsilon
FRAME_PER_ACTION = 1
INITIAL_EPSILON = 0.06 #greedy exploration
FINAL_EPSILON = 0.05
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100000 # timesteps to observe before training
REPLAY_MEMORY = 100000 # number of previous transitions to remember
BATCH = 32# size of minibatch


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)
def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

class double_DQN():
    def __init__(self):
        #卷积层的参数，为8*8的卷积核，输入4个通道，输出32个通道
        self.W_conv1 = weight_variable([8,8,4,32])
        self.b_conv1 = bias_variable([32])

        self.W_conv2 = weight_variable([4, 4, 32, 64])
        self.b_conv2 = bias_variable([64])

        self.W_conv3 = weight_variable([3, 3, 64, 64])
        self.b_conv3 = bias_variable([64])

        #全连接层的参数
        self.W_fc1 = weight_variable([1600, 512])
        self.b_fc1 = bias_variable([512])

        self.W_fc2 = weight_variable([512, ACTIONS])
        self.b_fc2 = bias_variable([ACTIONS])

        #输入层;每4帧一起输入
        self.s = tf.placeholder(tf.float32,[None,80,80,4])
        #隐藏层
        self.h_conv1 = tf.nn.relu(conv2d(self.s, self.W_conv1, 4) + self.b_conv1)
        self.h_pool1 = max_pool_2x2(self.h_conv1)

        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2, 2) + self.b_conv2)

        self.h_conv3 = tf.nn.relu(conv2d(self.h_conv2, self.W_conv3, 1) + self.b_conv3)
        #reshape为1*1600
        self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 1600])
        #shape is 1*512
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc1) + self.b_fc1)

        #输出层
        #shape is 1*ACTIONS
        self.readout = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2
        self.predict = tf.argmax(self.readout, 1) #选择reward最大的action

        self.a = tf.placeholder(tf.float32, [None, ACTIONS])
        #实际的reward,也就是q-target的值
        self.y = tf.placeholder(tf.float32, [None])

        #预期获得的reward
        self.readout_action = tf.reduce_sum(tf.multiply(self.readout, self.a), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y - self.readout_action))
        self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)


def trainNetwork(target_q,current_q,sess):

    game_state = t2.game()

    # store the previous observations in replay memory
    # deque 双向队列，D,E,F分别表示三类游戏情况
    D = deque()
    E = deque()
    F = deque()

    #日志
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    #游戏一开始，状态初始化。游戏画面为80*80，将第一帧重复4次作为原始网络的输入
    do_nothing = np.zeros(ACTIONS)
    do_nothing[random.randrange(ACTIONS)] = 1

    t = 0 #计数器，记录observation数目
    flag = np.zeros(3)

    #返回值分别为游戏画面，奖励，终端(判定游戏是否已结束，结束为True),树莓的位置，蛇的位置,以及此时蛇头的方向
    x_t, r_0, terminal, flag,raspberryPosition, snakePosition, direction= game_state.frame_step(do_nothing)
    #将游戏画面(原始为300*500)resize为(80*80)，以适应在配置较差的设备上运行。这样会损失训练效果。
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)

    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    update_w1=tf.assign(target_q.W_conv1,current_q.W_conv1)
    update_h1=tf.assign(target_q.b_conv1,current_q.b_conv1)
    update_w2=tf.assign(target_q.W_conv2,current_q.W_conv2)
    update_h2=tf.assign(target_q.b_conv2,current_q.b_conv2)
    update_w3=tf.assign(target_q.W_conv3,current_q.W_conv3)
    update_h3=tf.assign(target_q.b_conv3,current_q.b_conv3)
    update_w_fc1=tf.assign(target_q.W_fc1,current_q.W_fc1)
    update_b_fc1=tf.assign(target_q.b_fc1,current_q.b_fc1)
    update_w_fc2=tf.assign(target_q.W_fc2,current_q.W_fc2)
    update_b_fc2=tf.assign(target_q.b_fc2,current_q.b_fc2)
    sess.run(tf.global_variables_initializer())

    #保存与加载模型
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    saver = tf.train.Saver()
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    epsilon = INITIAL_EPSILON
    while(True):

        #greedy exploration
        readout_t = current_q.readout.eval(feed_dict={current_q.s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        #if t % FRAME_PER_ACTION == 0:
        #最开始人工指导snake运动,判断如果向左向右向上向下运动会怎么样
        if t<OBSERVE and not snakePosition==raspberryPosition and t%2==0:
            #dis是当前距离树莓派的位置
            dis=np.sqrt(np.square(snakePosition[0]-raspberryPosition[0])+np.square(snakePosition[1]-raspberryPosition[1]))
            #disl,disr,disu,disd分别是如果往左右上下运动后距离树莓派的距离
            disr=np.sqrt(np.square((snakePosition[0]+20)-raspberryPosition[0])+np.square(snakePosition[1]-raspberryPosition[1]))
            disl=np.sqrt(np.square((snakePosition[0]-20)-raspberryPosition[0])+np.square(snakePosition[1]-raspberryPosition[1]))
            disu=np.sqrt(np.square((snakePosition[0])-raspberryPosition[0])+np.square((snakePosition[1]-20)-raspberryPosition[1]))
            disd=np.sqrt(np.square((snakePosition[0])-raspberryPosition[0])+np.square((snakePosition[1]+20)-raspberryPosition[1]))
            Dist = [disr-dis,disl-dis,disu-dis,disd-dis]
            action_index = Dist.index(min(Dist))
            #蛇不可以往回走，所以最优的走法冲突，重新选择次优走法
            if action_index == 0 and direction=='left':
                Dist[action_index]=100
                action_index = Dist.index(min(Dist))
            elif action_index == 1 and direction=='right':
                Dist[action_index]=100
                action_index = Dist.index(min(Dist))
            elif action_index == 2 and direction=='down':
                Dist[action_index]=100
                action_index = Dist.index(min(Dist))
            elif action_index == 3 and direction=='up':
                Dist[action_index]=100
                action_index = Dist.index(min(Dist))
            a_t[action_index] = 1

        elif random.random() <= epsilon:
            #print("----------Random Action----------")
            action_index = random.randrange(ACTIONS)
            a_t[random.randrange(ACTIONS)] = 1
        else:
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1

        #scale the epsilon
        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action to observe next state and reward
        x_t1_colored, r_t, terminal, flag,raspberryPosition,snakePosition,direction = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        D.append((s_t, a_t, r_t, s_t1, terminal))
        if flag[1]==1:
            E.append((s_t, a_t, r_t, s_t1, terminal))
        elif flag[2]==1:
            F.append((s_t, a_t, r_t, s_t1, terminal))

        #如果蛇死亡，重新开始游戏
        if terminal:
            #s_t=copy.copy(s_temp)
            x_t, r_0, terminal,flag,raspberryPosition,snakePosition,direction= game_state.frame_step(do_nothing)
            x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
            ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
            s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        else:
            s_t=s_t1
        if len(D) > REPLAY_MEMORY:
            D.popleft()
        if len(E) > REPLAY_MEMORY:
            E.popleft()
        if len(F) > REPLAY_MEMORY:
            F.popleft()

        # only train if observation is done
        if t > OBSERVE:
            if(t>300000):
                minibatch = random.sample(D,8)
                minibatch+= random.sample(E,8)
                minibatch+= random.sample(F,16)
            else:
                minibatch = random.sample(D,24)
                minibatch+= random.sample(E,8)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            # y_batch = []
            #挑选action
            A = current_q.predict.eval(feed_dict = {current_q.s : s_j1_batch})
            #评价action的reward
            Q = target_q.readout.eval(feed_dict = {target_q.s : s_j1_batch})
            doubleQ = Q[range(32), A]

            #上一状态的Q值加上动作的reward等于当前的Q值
            #Qt+1 = r_t + gamma*Qt
            doubleQ=r_batch + GAMMA *doubleQ
            current_q.train_step.run(feed_dict = {
                    current_q.y : doubleQ,
                    current_q.a : a_batch,
                    current_q.s : s_j_batch}
            )

            if t %500==0:
                sess.run(update_w1)
                sess.run(update_w2)
                sess.run(update_w3)
                sess.run(update_h1)
                sess.run(update_h2)
                sess.run(update_h3)
                sess.run(update_w_fc1)
                sess.run(update_w_fc2)
                sess.run(update_b_fc1)
                sess.run(update_b_fc2)
        t += 1

        #save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)
            print("-----done steps:",t,'epsilon is:',epsilon,'-----')

def playGame():
    target_q= double_DQN()
    current_q= double_DQN()
    sess = tf.InteractiveSession()

#target_q= createNetwork()

    trainNetwork(target_q,current_q,sess)

def main():
    playGame()

if __name__ == "__main__":
    main()