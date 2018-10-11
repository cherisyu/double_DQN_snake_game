# coding=UTF-8
#!/usr/bin/env python
import pygame,sys,time,random
from pygame.locals import *
import numpy as np
import copy
#定义颜色变量
#redColour = pygame.Color(255,255,255)
blackColour = pygame.Color(0,0,0)
whiteColour = pygame.Color(255,255,255)
greyColour = pygame.Color(255,0,0)

# 定义gameOver函数
class game:



    def __init__(self):

        # 初始化pygame
        # pygame.init()
        pygame.display.init()
        pygame.font.init()
        pygame.joystick.init()
        self.fpsClock = pygame.time.Clock()

         # 创建pygame显示层
        self.playSurface = pygame.display.set_mode((300,500))
        pygame.display.set_caption('Raspberry Snake')
        #print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")

        # 初始化变量
        self.snakePosition = [140,240]
        #temp2=self.snakePosition
        #self.temp=temp2
        self.temp=copy.copy(self.snakePosition)
        self.snakeSegments = [[140,240]]
        x = random.randrange(0,15)
        y = random.randrange(0,25)
        self.raspberryPosition = [int(x*20),int(y*20)]
        self.raspberrySpawned = 1
        a=random.randint(0,3)
        if a==0:
            self.direction = 'right'
        if a==1:
            self.direction = 'left'
        if a==2:
            self.direction = 'up'
        if a==3:
            self.direction = 'down'

        self.changeDirection = self.direction
        self.flag=np.zeros(3)



    def frame_step(self,input_actions):
        q=0 #标志位，表示刚刚生成树莓派是不是立刻被吃掉
        reward=0
        terminal=False
        temp=copy.copy(self.snakePosition)

        self.flag[0]=0 #为1表示当前的动作向树莓派靠近了，或者远离了
        self.flag[1]=0 #为1表示当前的动作吃到了树莓派
        self.flag[2]=0 #为1表示当前的动作使蛇死亡



        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        if input_actions[0]==1:
            self.changeDirection = 'right'
        if input_actions[1]==1:
            self.changeDirection = 'left'
        if input_actions[2]==1:
            self.changeDirection = 'up'
        if input_actions[3]==1:
            self.changeDirection = 'down'

        #蛇不可以往回走
        if input_actions[0]==1 and not self.direction == 'left':
            self.direction = self.changeDirection
        if input_actions[1]==1 and not self.direction == 'right':
            self.direction = self.changeDirection
        if input_actions[2]==1 and not self.direction == 'down':
            self.direction = self.changeDirection
        if input_actions[3]==1 and not self.direction == 'up':
            self.direction = self.changeDirection
        # 根据方向移动蛇头的坐标
        if self.direction == 'right':
            self.snakePosition[0] += 20

        if self.direction == 'left':
            self.snakePosition[0] -= 20

        if self.direction == 'up':
            self.snakePosition[1] -= 20

        if self.direction == 'down':
            self.snakePosition[1] += 20

        # 增加蛇的长度,同时也实现了蛇的移动
        self.snakeSegments.insert(0,list(self.snakePosition))

        # 判断是否吃掉了树莓
        if self.snakePosition[0] == self.raspberryPosition[0] and self.snakePosition[1] == self.raspberryPosition[1]:
            self.raspberrySpawned = 0
            self.flag[0]=0
            self.flag[1]=1
            self.flag[2]=0
        else:
            self.snakeSegments.pop()
        # 如果吃掉树莓，则重新生成树莓
        if self.raspberrySpawned == 0:
            while(True):
                x = random.randrange(0,15)
                y = random.randrange(0,25)
                self.raspberryPosition = [int(x*20),int(y*20)]
                for position in self.snakeSegments:
                    if position==self.raspberryPosition:
                        q=1
                if q==1:
                    q=0
                    continue
                else:
                    break
            self.raspberrySpawned = 1

        #判断此时蛇头与树莓之间的距离，如果缩小就给予正奖励，否则负奖励
        dis1=np.sqrt(np.square(self.snakePosition[0]-self.raspberryPosition[0])+np.square(self.snakePosition[1]-self.raspberryPosition[1]))
        dis2=np.sqrt(np.square(temp[0]-self.raspberryPosition[0])+np.square(temp[1]-self.raspberryPosition[1]))
        if dis2>dis1:
            reward=1
            self.flag[0]=1
            self.flag[1]=0
            self.flag[2]=0
        elif dis2<dis1:
            reward=-1
            self.flag[0]=1
            self.flag[1]=0
            self.flag[2]=0

        #绘制pygame的显示层
        self.playSurface.fill(blackColour)

        for position in self.snakeSegments:
            pygame.draw.rect(self.playSurface,whiteColour,Rect(position[0],position[1],20,20))
            pygame.draw.rect(self.playSurface,greyColour,Rect(self.raspberryPosition[0], self.raspberryPosition[1],20,20))
        pygame.display.flip()
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())


        #判断蛇是不是死了
        if self.snakePosition[0] > 280 or self.snakePosition[0] < 0:
            terminal=True
            reward = -10
            self.flag[0]=0
            self.flag[1]=0
            self.flag[2]=1
            self.__init__()

        if self.snakePosition[1] > 480 or self.snakePosition[1] < 0:
            terminal=True
            reward = -10
            self.flag[0]=0
            self.flag[1]=0
            self.flag[2]=1
            self.__init__()

        for snakeBody in self.snakeSegments[1:]:
            if self.snakePosition[0] == snakeBody[0] and self.snakePosition[1] == snakeBody[1]:
                terminal = True
                reward1=-10
                self.flag[0]=0
                self.flag[1]=0
                self.flag[2]=1
                self.__init__()




        pygame.display.update()
        # 控制游戏速度
        self.fpsClock.tick(80)
        return image_data, reward,terminal,self.flag,self.raspberryPosition,self.snakePosition,self.direction


