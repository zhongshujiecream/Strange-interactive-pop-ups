import pygame
import random
import math
import sys

# 初始化pygame
pygame.init()

# 设置窗口
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3D梦境生成器")

# 颜色定义
COLORS = [
    (138, 43, 226),  # 蓝紫色
    (75, 0, 130),    # 靛蓝色
    (0, 191, 255),   # 深天蓝色
    (255, 20, 147),  # 深粉色
    (0, 255, 255),   # 青色
    (255, 105, 180), # 热粉色
    (147, 112, 219), # 中紫色
    (72, 61, 139),   # 暗紫罗兰色
]

# 3D立方体类
class DreamCube:
    def __init__(self):
        self.x = random.randint(-WIDTH//2, WIDTH//2)
        self.y = random.randint(-HEIGHT//2, HEIGHT//2)
        self.z = random.randint(100, 500)  # z坐标创造深度
        self.size = random.randint(20, 60)
        self.color = random.choice(COLORS)
        self.rotation_x = random.uniform(0, 2 * math.pi)
        self.rotation_y = random.uniform(0, 2 * math.pi)
        self.rotation_z = random.uniform(0, 2 * math.pi)
        self.rotation_speed_x = random.uniform(-0.02, 0.02)
        self.rotation_speed_y = random.uniform(-0.02, 0.02)
        self.rotation_speed_z = random.uniform(-0.02, 0.02)
        self.speed_z = random.uniform(-0.5, 0.5)
        
    def update(self, mouse_x, mouse_y):
        # 鼠标影响旋转
        self.rotation_speed_x += (mouse_y - HEIGHT//2) * 0.00001
        self.rotation_speed_y += (mouse_x - WIDTH//2) * 0.00001
        
        # 更新旋转
        self.rotation_x += self.rotation_speed_x
        self.rotation_y += self.rotation_speed_y
        self.rotation_z += self.rotation_speed_z
        
        # 更新z坐标
        self.z += self.speed_z
        if self.z < 50 or self.z > 800:
            self.speed_z *= -1
            
        # 边界检查
        if abs(self.x) > WIDTH//2:
            self.x = -self.x * 0.8
        if abs(self.y) > HEIGHT//2:
            self.y = -self.y * 0.8
            
    def project_3d_to_2d(self, x, y, z):
        """将3D坐标投影到2D屏幕"""
        scale = 500 / (z + 500)  # 透视投影
        screen_x = WIDTH//2 + x * scale
        screen_y = HEIGHT//2 + y * scale
        return screen_x, screen_y, scale
        
    def draw(self, surface):
        # 定义立方体的8个顶点
        vertices = []
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    vertices.append((x * self.size, y * self.size, z * self.size))
        
        # 旋转顶点
        rotated_vertices = []
        for x, y, z in vertices:
            # 绕Y轴旋转
            y_rotated = y * math.cos(self.rotation_y) - z * math.sin(self.rotation_y)
            z_rotated = y * math.sin(self.rotation_y) + z * math.cos(self.rotation_y)
            
            # 绕X轴旋转
            x_rotated = x * math.cos(self.rotation_x) + z_rotated * math.sin(self.rotation_x)
            z_final = -x * math.sin(self.rotation_x) + z_rotated * math.cos(self.rotation_x)
            
            # 绕Z轴旋转
            x_final = x_rotated * math.cos(self.rotation_z) - y_rotated * math.sin(self.rotation_z)
            y_final = x_rotated * math.sin(self.rotation_z) + y_rotated * math.cos(self.rotation_z)
            
            rotated_vertices.append((x_final, y_final, z_final))
        
        # 投影到2D并绘制
        projected_vertices = []
        for x, y, z in rotated_vertices:
            screen_x, screen_y, scale = self.project_3d_to_2d(
                self.x + x, self.y + y, self.z + z
            )
            projected_vertices.append((screen_x, screen_y, scale))
        
        # 绘制边
        edges = [
            (0,1), (1,3), (3,2), (2,0),  # 前面
            (4,5), (5,7), (7,6), (6,4),  # 后面
            (0,4), (1,5), (2,6), (3,7)   # 连接线
        ]
        
        for start, end in edges:
            x1, y1, scale1 = projected_vertices[start]
            x2, y2, scale2 = projected_vertices[end]
            
            # 根据深度调整颜色透明度
            avg_scale = (scale1 + scale2) / 2
            alpha = int(255 * avg_scale)
            color_with_alpha = (*self.color, alpha)
            
            # 绘制线条
            line_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            pygame.draw.line(line_surface, color_with_alpha, (x1, y1), (x2, y2), 2)
            surface.blit(line_surface, (0, 0))

# 浮动建筑类
class FloatingBuilding:
    def __init__(self):
        self.x = random.randint(-WIDTH//2, WIDTH//2)
        self.y = random.randint(-HEIGHT//2, HEIGHT//2)
        self.z = random.randint(200, 600)
        self.width = random.randint(30, 80)
        self.height = random.randint(50, 150)
        self.depth = random.randint(20, 60)
        self.color = random.choice(COLORS)
        self.speed_z = random.uniform(-0.3, 0.3)
        self.window_color = (255, 255, 200)
        
    def update(self, mouse_x, mouse_y):
        # 鼠标影响移动
        self.x += (mouse_x - WIDTH//2) * 0.001
        self.y += (mouse_y - HEIGHT//2) * 0.001
        
        # z轴移动
        self.z += self.speed_z
        if self.z < 100 or self.z > 800:
            self.speed_z *= -1
            
    def project_3d_to_2d(self, x, y, z):
        scale = 500 / (z + 500)
        screen_x = WIDTH//2 + x * scale
        screen_y = HEIGHT//2 + y * scale
        return screen_x, screen_y, scale
        
    def draw(self, surface):
        # 建筑的主要顶点
        vertices = [
            (-self.width/2, -self.height/2, -self.depth/2),
            (self.width/2, -self.height/2, -self.depth/2),
            (self.width/2, self.height/2, -self.depth/2),
            (-self.width/2, self.height/2, -self.depth/2),
            (-self.width/2, -self.height/2, self.depth/2),
            (self.width/2, -self.height/2, self.depth/2),
            (self.width/2, self.height/2, self.depth/2),
            (-self.width/2, self.height/2, self.depth/2)
        ]
        
        # 投影顶点
        projected = []
        for x, y, z in vertices:
            screen_x, screen_y, scale = self.project_3d_to_2d(
                self.x + x, self.y + y, self.z + z
            )
            projected.append((screen_x, screen_y, scale))
        
        # 绘制建筑轮廓
        edges = [
            (0,1), (1,2), (2,3), (3,0),  # 前面
            (4,5), (5,6), (6,7), (7,4),  # 后面
            (0,4), (1,5), (2,6), (3,7)   # 连接线
        ]
        
        for start, end in edges:
            x1, y1, scale1 = projected[start]
            x2, y2, scale2 = projected[end]
            avg_scale = (scale1 + scale2) / 2
            alpha = int(200 * avg_scale)
            color_with_alpha = (*self.color, alpha)
            
            line_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            pygame.draw.line(line_surface, color_with_alpha, (x1, y1), (x2, y2), 3)
            surface.blit(line_surface, (0, 0))
        
        # 绘制窗户
        window_size = max(2, int(5 * projected[0][2]))
        for i in range(3):  # 3层窗户
            for j in range(2):  # 2列窗户
                window_x = self.x - self.width/4 + j * self.width/2
                window_y = self.y - self.height/3 + i * self.height/3
                window_z = self.z - self.depth/2 - 5
                
                wx, wy, scale = self.project_3d_to_2d(window_x, window_y, window_z)
                if random.random() < 0.7:  # 70%的窗户亮着
                    pygame.draw.rect(surface, self.window_color, 
                                   (wx - window_size//2, wy - window_size//2, 
                                    window_size, window_size))

# 浮动球体类
class FloatingSphere:
    def __init__(self):
        self.x = random.randint(-WIDTH//2, WIDTH//2)
        self.y = random.randint(-HEIGHT//2, HEIGHT//2)
        self.z = random.randint(100, 400)
        self.radius = random.randint(15, 40)
        self.color = random.choice(COLORS)
        self.speed_x = random.uniform(-1, 1)
        self.speed_y = random.uniform(-1, 1)
        self.speed_z = random.uniform(-0.5, 0.5)
        
    def update(self, mouse_x, mouse_y):
        # 鼠标排斥效果
        dx = self.x - (mouse_x - WIDTH//2)
        dy = self.y - (mouse_y - HEIGHT//2)
        distance = max(1, math.sqrt(dx*dx + dy*dy))
        
        if distance < 200:
            force = (200 - distance) / 200
            self.speed_x += (dx / distance) * force * 0.5
            self.speed_y += (dy / distance) * force * 0.5
        
        # 更新位置
        self.x += self.speed_x
        self.y += self.speed_y
        self.z += self.speed_z
        
        # 边界检查
        if abs(self.x) > WIDTH//2:
            self.speed_x *= -0.8
        if abs(self.y) > HEIGHT//2:
            self.speed_y *= -0.8
        if self.z < 50 or self.z > 600:
            self.speed_z *= -0.8
            
    def draw(self, surface):
        scale = 500 / (self.z + 500)
        screen_x = WIDTH//2 + self.x * scale
        screen_y = HEIGHT//2 + self.y * scale
        screen_radius = int(self.radius * scale)
        
        # 绘制球体
        sphere_surface = pygame.Surface((screen_radius*2, screen_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(sphere_surface, (*self.color, 180), 
                         (screen_radius, screen_radius), screen_radius)
        
        # 添加高光
        highlight_pos = (screen_radius//2, screen_radius//2)
        highlight_radius = screen_radius//3
        pygame.draw.circle(sphere_surface, (255, 255, 255, 100), 
                         highlight_pos, highlight_radius)
        
        surface.blit(sphere_surface, (screen_x - screen_radius, screen_y - screen_radius))

# 隧道类
class DreamTunnel:
    def __init__(self):
        self.segments = []
        self.generate_tunnel()
        
    def generate_tunnel(self):
        # 生成隧道段
        for i in range(20):
            radius = 100 - i * 3  # 逐渐变小的隧道
            color = random.choice(COLORS)
            self.segments.append({
                'z': i * 50,
                'radius': radius,
                'color': color,
                'rotation': random.uniform(0, 2 * math.pi)
            })
            
    def update(self, mouse_x, mouse_y):
        # 更新隧道旋转
        for segment in self.segments:
            segment['rotation'] += (mouse_x - WIDTH//2) * 0.0001
            segment['z'] -= 2  # 隧道向前移动
            
        # 循环隧道
        if self.segments[0]['z'] < -50:
            segment = self.segments.pop(0)
            segment['z'] = self.segments[-1]['z'] + 50
            self.segments.append(segment)
            
    def draw(self, surface):
        for segment in self.segments:
            z = segment['z']
            if z > 0:  # 只绘制在屏幕内的部分
                scale = 500 / (z + 500)
                radius = int(segment['radius'] * scale)
                
                # 绘制隧道环
                tunnel_surface = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                center = (radius, radius)
                
                # 绘制多个点形成环形
                points = []
                for i in range(12):
                    angle = segment['rotation'] + i * math.pi / 6
                    x = radius + math.cos(angle) * radius
                    y = radius + math.sin(angle) * radius
                    points.append((x, y))
                
                # 连接点形成隧道环
                for i in range(len(points)):
                    start = points[i]
                    end = points[(i + 1) % len(points)]
                    pygame.draw.line(tunnel_surface, (*segment['color'], 150), 
                                   start, end, 3)
                
                surface.blit(tunnel_surface, 
                           (WIDTH//2 - radius, HEIGHT//2 - radius))

# 创建各种3D元素
cubes = [DreamCube() for _ in range(8)]
buildings = [FloatingBuilding() for _ in range(6)]
spheres = [FloatingSphere() for _ in range(12)]
tunnel = DreamTunnel()

# 主循环
clock = pygame.time.Clock()
running = True
mouse_x, mouse_y = WIDTH//2, HEIGHT//2

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEMOTION:
            mouse_x, mouse_y = event.pos
    
    # 创建半透明背景，产生拖尾效果
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(20)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))
    
    # 绘制隧道（作为背景）
    tunnel.update(mouse_x, mouse_y)
    tunnel.draw(screen)
    
    # 更新和绘制所有3D元素
    for cube in cubes:
        cube.update(mouse_x, mouse_y)
        cube.draw(screen)
    
    for building in buildings:
        building.update(mouse_x, mouse_y)
        building.draw(screen)
    
    for sphere in spheres:
        sphere.update(mouse_x, mouse_y)
        sphere.draw(screen)
    
    # 随机添加闪烁星星作为点缀
    if random.random() < 0.3:
        x = random.randint(0, WIDTH)
        y = random.randint(0, HEIGHT)
        size = random.randint(1, 2)
        pygame.draw.circle(screen, (255, 255, 255), (x, y), size)
    
    # 在鼠标位置绘制交互指示器
    pygame.draw.circle(screen, (255, 255, 255, 100), (mouse_x, mouse_y), 10, 2)
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()