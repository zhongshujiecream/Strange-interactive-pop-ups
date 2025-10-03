import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import random

# 页面配置
st.set_page_config(
    page_title="🎨 人脸纹身艺术效果器",
    page_icon="🎨",
    layout="wide"
)

# CSS样式
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .effect-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# 主标题
st.markdown('<div class="main-header"><h1>🎨 实时人脸纹身艺术效果器</h1><p>将您的摄像头画面转换为抽象艺术作品 + 智能人脸纹身</p></div>', unsafe_allow_html=True)

# 加载人脸检测器（使用OpenCV内置）
@st.cache_resource
def load_face_cascade():
    try:
        # 使用OpenCV内置的人脸检测器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return face_cascade
    except Exception as e:
        st.error(f"人脸检测器加载失败: {e}")
        return None

face_cascade = load_face_cascade()

# 侧边栏控制面板
with st.sidebar:
    st.header("🎛️ 效果控制面板")
    
    # 人脸纹身控制
    st.subheader("👤 人脸纹身设置")
    enable_face_tattoo = st.checkbox("启用人脸纹身", value=True)
    
    if enable_face_tattoo:
        tattoo_style = st.selectbox(
            "纹身样式",
            [
                "彩色花朵",
                "部落图腾", 
                "几何图案",
                "星座符号",
                "蝴蝶翅膀",
                "龙纹刺青",
                "玫瑰花环",
                "闪电符号"
            ]
        )
        
        tattoo_opacity = st.slider("纹身透明度", 0.1, 1.0, 0.4, 0.05)
        tattoo_size = st.slider("纹身大小", 0.3, 1.5, 0.8, 0.1)
        tattoo_color_mode = st.selectbox("颜色模式", ["随机彩色", "单色", "渐变"])
    
    st.markdown("---")
    
    # 抽象效果选择
    st.subheader("🎨 抽象背景效果")
    effect_type = st.selectbox(
        "选择抽象效果",
        [
            "原始画面",
            "油画效果", 
            "水彩画效果",
            "马赛克艺术",
            "边缘抽象",
            "色彩分离",
            "万花筒",
            "波普艺术",
            "数字抽象",
            "粒子效果",
            "镜像艺术",
            "色彩爆炸"
        ]
    )
    
    st.markdown("---")
    
    # 效果强度控制
    intensity = st.slider("效果强度", 1, 10, 5)
    
    # 颜色控制
    st.subheader("🎨 颜色调节")
    color_shift = st.slider("色彩偏移", 0, 100, 0)
    saturation = st.slider("饱和度", 0.5, 3.0, 1.0, 0.1)
    brightness = st.slider("亮度", 0.5, 2.0, 1.0, 0.1)
    
    # 动态效果
    st.subheader("⚡ 动态效果")
    enable_animation = st.checkbox("启用动画效果")
    animation_speed = st.slider("动画速度", 1, 5, 2) if enable_animation else 2
    
    # 将UI控件值存储到session_state供回调函数使用
    st.session_state.current_enable_face_tattoo = enable_face_tattoo
    if enable_face_tattoo:
        st.session_state.current_tattoo_style = tattoo_style
        st.session_state.current_tattoo_opacity = tattoo_opacity
        st.session_state.current_tattoo_size = tattoo_size
        st.session_state.current_tattoo_color_mode = tattoo_color_mode
    
    st.session_state.current_effect_type = effect_type
    st.session_state.current_intensity = intensity
    st.session_state.current_color_shift = color_shift
    st.session_state.current_saturation = saturation
    st.session_state.current_brightness = brightness
    st.session_state.current_enable_animation = enable_animation
    st.session_state.current_animation_speed = animation_speed
    
    # 随机化
    if st.button("🎲 随机效果"):
        st.session_state.random_effect = True
    
    st.markdown("---")
    st.markdown("**💡 使用提示:**")
    st.markdown("- 允许摄像头访问")
    st.markdown("- 面向摄像头以检测人脸")
    st.markdown("- 尝试不同纹身样式")
    st.markdown("- 调节参数获得最佳效果")

# 全局变量用于动画
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

def apply_oil_painting(img, intensity):
    """油画效果"""
    ksize = intensity * 2 + 1
    return cv2.bilateralFilter(img, 15, 80, 80)

def apply_watercolor(img, intensity):
    """水彩画效果"""
    img_blur = cv2.medianBlur(img, intensity * 3 + 5)
    img_edge = cv2.adaptiveThreshold(
        cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY), 
        255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10
    )
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(img_blur, img_edge)

def apply_mosaic(img, intensity):
    """马赛克艺术效果"""
    h, w = img.shape[:2]
    block_size = max(5, 20 - intensity * 2)
    
    small = cv2.resize(img, (w//block_size, h//block_size), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return mosaic

def apply_edge_abstract(img, intensity):
    """边缘抽象效果"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    colored_edges = np.zeros_like(img)
    colored_edges[:,:,0] = edges
    colored_edges[:,:,1] = edges // 2
    colored_edges[:,:,2] = edges
    
    alpha = intensity / 10.0
    return cv2.addWeighted(img, 1-alpha, colored_edges, alpha, 0)

def apply_color_separation(img, intensity):
    """色彩分离效果"""
    h, w = img.shape[:2]
    offset = intensity * 3
    
    result = np.zeros_like(img)
    
    if offset < w:
        result[:, offset:, 2] = img[:, :-offset, 2]
    
    result[:, :, 1] = img[:, :, 1]
    
    if offset < w:
        result[:, :-offset, 0] = img[:, offset:, 0]
    
    return result

def apply_kaleidoscope(img, intensity):
    """万花筒效果"""
    h, w = img.shape[:2]
    center = (w//2, h//2)
    
    angle = 360 // (intensity + 2)
    
    result = img.copy()
    for i in range(0, 360, angle):
        M = cv2.getRotationMatrix2D(center, i, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        result = cv2.addWeighted(result, 0.7, rotated, 0.3, 0)
    
    return result

def apply_pop_art(img, intensity):
    """波普艺术效果"""
    data = img.reshape((-1, 3))
    data = np.float32(data)
    
    k = max(3, 12 - intensity)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_img = segmented_data.reshape(img.shape)
    
    return segmented_img

def apply_digital_abstract(img, intensity):
    """数字抽象效果"""
    h, w = img.shape[:2]
    grid_size = max(10, 50 - intensity * 4)
    
    for i in range(0, h, grid_size):
        for j in range(0, w, grid_size):
            color = [random.randint(0, 255) for _ in range(3)]
            cv2.rectangle(img, (j, i), (j+grid_size//2, i+grid_size//2), color, -1)
    
    return img

def apply_particle_effect(img, intensity):
    """粒子效果"""
    h, w = img.shape[:2]
    particles = intensity * 100
    
    result = img.copy()
    for _ in range(particles):
        x = random.randint(0, w-1)
        y = random.randint(0, h-1)
        radius = random.randint(1, intensity)
        color = [random.randint(0, 255) for _ in range(3)]
        cv2.circle(result, (x, y), radius, color, -1)
    
    return cv2.addWeighted(img, 0.7, result, 0.3, 0)

def apply_mirror_art(img, intensity):
    """镜像艺术效果"""
    h, w = img.shape[:2]
    
    if intensity % 4 == 1:
        left_half = img[:, :w//2]
        result = np.hstack([left_half, cv2.flip(left_half, 1)])
    elif intensity % 4 == 2:
        top_half = img[:h//2, :]
        result = np.vstack([top_half, cv2.flip(top_half, 0)])
    elif intensity % 4 == 3:
        quarter = img[:h//2, :w//2]
        top = np.hstack([quarter, cv2.flip(quarter, 1)])
        bottom = cv2.flip(top, 0)
        result = np.vstack([top, bottom])
    else:
        result = cv2.flip(cv2.flip(img, 0), 1)
    
    return result

def apply_color_explosion(img, intensity):
    """色彩爆炸效果"""
    h, w = img.shape[:2]
    center = (w//2, h//2)
    
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    
    max_dist = np.sqrt(center[0]**2 + center[1]**2)
    mask = (dist_from_center / max_dist) * intensity
    
    result = img.copy().astype(np.float32)
    result[:,:,0] *= (1 + mask * 0.5)
    result[:,:,1] *= (1 + mask * 0.3)
    result[:,:,2] *= (1 + mask * 0.7)
    
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_color_adjustments(img, color_shift, saturation, brightness):
    """应用颜色调整"""
    if color_shift > 0:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,0] = (hsv[:,:,0] + color_shift) % 180
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    img = img.astype(np.float32)
    img = img * brightness
    
    if saturation != 1.0:
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] = hsv[:,:,1] * saturation
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    
    return np.clip(img, 0, 255).astype(np.uint8)

def draw_face_tattoo(img, face_rect, style, opacity, size_factor, color_mode):
    """在人脸上绘制纹身"""
    x, y, w, h = face_rect
    center = (x + w//2, y + h//2)
    
    # 创建覆盖层
    overlay = img.copy()
    
    # 根据纹身大小调整半径
    base_radius = int(min(w, h) * 0.15 * size_factor)
    
    # 根据颜色模式设置颜色
    if color_mode == "随机彩色":
        color1 = (random.randint(100, 255), random.randint(50, 200), random.randint(100, 255))
        color2 = (random.randint(50, 200), random.randint(100, 255), random.randint(100, 255))
        color3 = (random.randint(100, 255), random.randint(100, 255), random.randint(50, 200))
    elif color_mode == "单色":
        base_color = random.randint(100, 255)
        color1 = color2 = color3 = (base_color, base_color//2, base_color//3)
    else:  # 渐变
        color1 = (255, 100, 150)
        color2 = (150, 255, 100)
        color3 = (100, 150, 255)
    
    if style == "彩色花朵":
        # 绘制花朵中心
        cv2.circle(overlay, center, base_radius//2, color1, -1)
        # 绘制花瓣
        for i in range(8):
            angle = i * np.pi / 4
            petal_x = int(center[0] + base_radius * np.cos(angle))
            petal_y = int(center[1] + base_radius * np.sin(angle))
            cv2.circle(overlay, (petal_x, petal_y), base_radius//3, color2, -1)
    
    elif style == "部落图腾":
        # 绘制部落风格的几何图案
        pts = []
        for i in range(6):
            angle = i * np.pi / 3
            pt_x = int(center[0] + base_radius * np.cos(angle))
            pt_y = int(center[1] + base_radius * np.sin(angle))
            pts.append([pt_x, pt_y])
        pts = np.array(pts, np.int32)
        cv2.fillPoly(overlay, [pts], color1)
        cv2.circle(overlay, center, base_radius//3, color2, -1)
    
    elif style == "几何图案":
        # 绘制几何三角形图案
        for i in range(3):
            angle = i * 2 * np.pi / 3
            pt1_x = int(center[0] + base_radius * np.cos(angle))
            pt1_y = int(center[1] + base_radius * np.sin(angle))
            pt2_x = int(center[0] + base_radius * np.cos(angle + np.pi/3))
            pt2_y = int(center[1] + base_radius * np.sin(angle + np.pi/3))
            cv2.line(overlay, center, (pt1_x, pt1_y), color1, 3)
            cv2.line(overlay, (pt1_x, pt1_y), (pt2_x, pt2_y), color2, 2)
    
    elif style == "星座符号":
        # 绘制星座样式的星形图案
        for i in range(5):
            angle1 = i * 2 * np.pi / 5
            angle2 = (i + 2) % 5 * 2 * np.pi / 5
            pt1_x = int(center[0] + base_radius * np.cos(angle1))
            pt1_y = int(center[1] + base_radius * np.sin(angle1))
            pt2_x = int(center[0] + base_radius * np.cos(angle2))
            pt2_y = int(center[1] + base_radius * np.sin(angle2))
            cv2.line(overlay, (pt1_x, pt1_y), (pt2_x, pt2_y), color1, 2)
        cv2.circle(overlay, center, base_radius//4, color2, -1)
    
    elif style == "蝴蝶翅膀":
        # 绘制蝴蝶翅膀图案
        wing_points = [
            (center[0] - base_radius, center[1] - base_radius//2),
            (center[0] - base_radius//2, center[1]),
            (center[0] - base_radius, center[1] + base_radius//2),
            (center[0] - base_radius//3, center[1])
        ]
        cv2.fillPoly(overlay, [np.array(wing_points, np.int32)], color1)
        
        wing_points_right = [
            (center[0] + base_radius, center[1] - base_radius//2),
            (center[0] + base_radius//2, center[1]),
            (center[0] + base_radius, center[1] + base_radius//2),
            (center[0] + base_radius//3, center[1])
        ]
        cv2.fillPoly(overlay, [np.array(wing_points_right, np.int32)], color2)
        cv2.circle(overlay, center, base_radius//6, color3, -1)
    
    elif style == "龙纹刺青":
        # 绘制龙形纹身
        cv2.ellipse(overlay, center, (base_radius, base_radius//2), 0, 0, 180, color1, -1)
        cv2.ellipse(overlay, center, (base_radius//2, base_radius), 45, 0, 180, color2, -1)
        cv2.circle(overlay, (center[0]-base_radius//3, center[1]-base_radius//3), base_radius//4, color3, -1)
    
    elif style == "玫瑰花环":
        # 绘制玫瑰花环
        for i in range(12):
            angle = i * np.pi / 6
            rose_x = int(center[0] + base_radius * 0.8 * np.cos(angle))
            rose_y = int(center[1] + base_radius * 0.8 * np.sin(angle))
            cv2.circle(overlay, (rose_x, rose_y), base_radius//6, color1, -1)
            cv2.circle(overlay, (rose_x, rose_y), base_radius//8, color2, -1)
    
    elif style == "闪电符号":
        # 绘制闪电图案
        pts = np.array([
            [center[0] - base_radius//3, center[1] - base_radius],
            [center[0] + base_radius//3, center[1] - base_radius//3],
            [center[0], center[1]],
            [center[0] + base_radius//2, center[1] + base_radius//3],
            [center[0] - base_radius//4, center[1] + base_radius],
            [center[0] - base_radius//6, center[1] + base_radius//3],
            [center[0] - base_radius//3, center[1]]
        ], np.int32)
        cv2.fillPoly(overlay, [pts], color1)
        cv2.polylines(overlay, [pts], True, color2, 2)
    
    # 应用透明度
    result = cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0)
    return result

def video_frame_callback(frame):
    """视频帧回调函数 - 应用所选效果 + 人脸纹身"""
    img = frame.to_ndarray(format="bgr24")
    
    # 增加帧计数用于动画
    st.session_state.frame_count += 1
    
    # 获取当前的UI控件值
    enable_face_tattoo = st.session_state.get('current_enable_face_tattoo', True)
    effect_type = st.session_state.get('current_effect_type', '原始画面')
    intensity = st.session_state.get('current_intensity', 5)
    color_shift = st.session_state.get('current_color_shift', 0)
    saturation = st.session_state.get('current_saturation', 1.0)
    brightness = st.session_state.get('current_brightness', 1.0)
    enable_animation = st.session_state.get('current_enable_animation', False)
    animation_speed = st.session_state.get('current_animation_speed', 2)
    
    if enable_face_tattoo:
        tattoo_style = st.session_state.get('current_tattoo_style', '彩色花朵')
        tattoo_opacity = st.session_state.get('current_tattoo_opacity', 0.4)
        tattoo_size = st.session_state.get('current_tattoo_size', 0.8)
        tattoo_color_mode = st.session_state.get('current_tattoo_color_mode', '随机彩色')
    
    # 检查是否启用随机效果
    if 'random_effect' in st.session_state:
        effects = ["油画效果", "水彩画效果", "马赛克艺术", "边缘抽象", "色彩分离", 
                  "万花筒", "波普艺术", "数字抽象", "粒子效果", "镜像艺术", "色彩爆炸"]
        effect_type = random.choice(effects)
        del st.session_state.random_effect
    
    # 动画效果调整
    if enable_animation:
        anim_intensity = intensity + int(3 * np.sin(st.session_state.frame_count * 0.1 * animation_speed))
        anim_intensity = max(1, min(10, anim_intensity))
    else:
        anim_intensity = intensity
    
    try:
        # 应用选择的抽象效果
        if effect_type == "油画效果":
            img = apply_oil_painting(img, anim_intensity)
        elif effect_type == "水彩画效果":
            img = apply_watercolor(img, anim_intensity)
        elif effect_type == "马赛克艺术":
            img = apply_mosaic(img, anim_intensity)
        elif effect_type == "边缘抽象":
            img = apply_edge_abstract(img, anim_intensity)
        elif effect_type == "色彩分离":
            img = apply_color_separation(img, anim_intensity)
        elif effect_type == "万花筒":
            img = apply_kaleidoscope(img, anim_intensity)
        elif effect_type == "波普艺术":
            img = apply_pop_art(img, anim_intensity)
        elif effect_type == "数字抽象":
            img = apply_digital_abstract(img, anim_intensity)
        elif effect_type == "粒子效果":
            img = apply_particle_effect(img, anim_intensity)
        elif effect_type == "镜像艺术":
            img = apply_mirror_art(img, anim_intensity)
        elif effect_type == "色彩爆炸":
            img = apply_color_explosion(img, anim_intensity)
        
        # 应用颜色调整
        img = apply_color_adjustments(img, color_shift, saturation, brightness)
        
        # ====== 人脸检测与纹身叠加 ======
        if enable_face_tattoo and face_cascade is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # 在每个检测到的人脸上添加纹身
            for (x, y, w, h) in faces:
                img = draw_face_tattoo(
                    img, 
                    (x, y, w, h), 
                    tattoo_style, 
                    tattoo_opacity, 
                    tattoo_size, 
                    tattoo_color_mode
                )
        
    except Exception as e:
        # 如果效果应用失败，返回原始图像
        pass
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 主要内容区域
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 实时视频流")
    
    # WebRTC流媒体组件
    webrtc_streamer(
        key="face_tattoo_effects",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

with col2:
    st.subheader("📊 效果信息")
    
    # 显示当前效果信息
    with st.container():
        st.markdown('<div class="effect-card">', unsafe_allow_html=True)
        st.write(f"**背景效果:** {effect_type}")
        st.write(f"**效果强度:** {intensity}/10")
        
        if enable_face_tattoo:
            st.write(f"**人脸纹身:** 启用")
            st.write(f"**纹身样式:** {tattoo_style}")
            st.write(f"**纹身透明度:** {tattoo_opacity:.2f}")
            st.write(f"**纹身大小:** {tattoo_size:.1f}x")
        else:
            st.write(f"**人脸纹身:** 禁用")
            
        st.write(f"**色彩偏移:** {color_shift}°")
        st.write(f"**饱和度:** {saturation:.1f}x")
        st.write(f"**亮度:** {brightness:.1f}x")
        if enable_animation:
            st.write(f"**动画:** 启用 (速度: {animation_speed})")
        else:
            st.write("**动画:** 禁用")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 人脸纹身预设
    st.subheader("👤 纹身快速预设")
    
    col_preset1, col_preset2 = st.columns(2)
    
    with col_preset1:
        if st.button("🌺 花朵女神"):
            st.session_state.update({
                'current_tattoo_style': '彩色花朵',
                'current_tattoo_opacity': 0.5,
                'current_tattoo_size': 1.0,
                'current_tattoo_color_mode': '随机彩色',
                'current_effect_type': '水彩画效果'
            })
        
        if st.button("⚡ 闪电战士"):
            st.session_state.update({
                'current_tattoo_style': '闪电符号',
                'current_tattoo_opacity': 0.7,
                'current_tattoo_size': 1.2,
                'current_tattoo_color_mode': '单色',
                'current_effect_type': '边缘抽象'
            })
        
        if st.button("🦋 蝴蝶仙子"):
            st.session_state.update({
                'current_tattoo_style': '蝴蝶翅膀',
                'current_tattoo_opacity': 0.4,
                'current_tattoo_size': 0.9,
                'current_tattoo_color_mode': '渐变',
                'current_effect_type': '粒子效果'
            })
    
    with col_preset2:
        if st.button("🐉 龙纹武士"):
            st.session_state.update({
                'current_tattoo_style': '龙纹刺青',
                'current_tattoo_opacity': 0.6,
                'current_tattoo_size': 1.3,
                'current_tattoo_color_mode': '渐变',
                'current_effect_type': '数字抽象'
            })
        
        if st.button("⭐ 星座法师"):
            st.session_state.update({
                'current_tattoo_style': '星座符号',
                'current_tattoo_opacity': 0.5,
                'current_tattoo_size': 0.8,
                'current_tattoo_color_mode': '随机彩色',
                'current_effect_type': '万花筒'
            })
        
        if st.button("🌹 玫瑰皇后"):
            st.session_state.update({
                'current_tattoo_style': '玫瑰花环',
                'current_tattoo_opacity': 0.45,
                'current_tattoo_size': 1.1,
                'current_tattoo_color_mode': '渐变',
                'current_effect_type': '色彩爆炸'
            })
    
    # 抽象效果预设
    st.subheader("🎨 抽象效果预设")
    
    col_effect1, col_effect2 = st.columns(2)
    
    with col_effect1:
        if st.button("🌈 彩虹梦境"):
            st.session_state.update({
                'current_effect_type': '色彩爆炸',
                'current_intensity': 8,
                'current_color_shift': 50,
                'current_saturation': 2.0,
                'current_enable_animation': True
            })
        
        if st.button("🎭 艺术大师"):
            st.session_state.update({
                'current_effect_type': '油画效果',
                'current_intensity': 6,
                'current_saturation': 1.5,
                'current_brightness': 1.2
            })
    
    with col_effect2:
        if st.button("🔮 神秘水晶"):
            st.session_state.update({
                'current_effect_type': '万花筒',
                'current_intensity': 7,
                'current_color_shift': 30,
                'current_enable_animation': True
            })
        
        if st.button("🎯 像素世界"):
            st.session_state.update({
                'current_effect_type': '马赛克艺术',
                'current_intensity': 4,
                'current_saturation': 2.5,
                'current_brightness': 1.3
            })
    
    if st.button("🔄 重置所有", type="primary", use_container_width=True):
        keys_to_reset = [
            'current_enable_face_tattoo', 'current_tattoo_style', 'current_tattoo_opacity',
            'current_tattoo_size', 'current_tattoo_color_mode', 'current_effect_type', 
            'current_intensity', 'current_color_shift', 'current_saturation', 
            'current_brightness', 'current_enable_animation'
        ]
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# 底部信息
st.markdown("---")
col_info1, col_info2, col_info3, col_info4 = st.columns(4)

with col_info1:
    st.markdown("**🛠️ 技术栈**")
    st.markdown("• Streamlit WebRTC")
    st.markdown("• OpenCV")
    st.markdown("• NumPy")

with col_info2:
    st.markdown("**🎨 抽象效果**")
    st.markdown("• 12种背景效果")
    st.markdown("• 实时颜色调节")
    st.markdown("• 动画支持")

with col_info3:
    st.markdown("**👤 人脸纹身**")
    st.markdown("• 8种纹身样式")
    st.markdown("• 实时人脸检测")
    st.markdown("• 多种颜色模式")

with col_info4:
    st.markdown("**💡 使用提示**")
    st.markdown("• 确保良好的光线")
    st.markdown("• 面向摄像头")
    st.markdown("• 尝试不同组合")