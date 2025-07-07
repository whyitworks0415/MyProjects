################################################################################
# bouncing_objects_optimized.py
#
# 설명:
#   - 정20면체 내부에서 100개의 구가 중력에 의해 낙하하며
#     면(삼각형)과 서로 충돌해 튕기는 시뮬레이션.
#   - GPU 부하를 줄이기 위해 “Sphere Display List”를 사용.
#   - CPU 부하를 줄이기 위해 “공간 분할(Spatial Hash Grid)”을 적용한 충돌 검사.
#
# 필요 라이브러리:
#   pip install PyOpenGL PyOpenGL_accelerate glfw numpy
#
# 실행:
#   python bouncing_objects_optimized.py
#
# 조작 방법:
#   - 마우스 왼쪽 버튼 드래그 : 카메라 회전 (드래그 방향의 반대 방향)
#   - 스크롤 휠               : 카메라 줌 인/아웃
#   - W / S                   : 카메라 앞뒤 이동 (타깃과의 거리 조절)
#   - A / D                   : 카메라 좌우 회전 (Yaw 축)
#   - R / F                   : 타깃 상하 이동 (Y 축)
#   - ↑ / ↓ (UP/DOWN)         : 카메라 피치 세밀 조정
#   - Space                   : 일시정지/재개
#   - Esc / Q                 : 종료
#
################################################################################

import sys
import random
import math
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *

# ───────── 시뮬레이션 파라미터 ─────────
NUM_SPHERES = 100       # 공 개수
RADIUS      = 0.3       # 공 반지름
DT          = 0.05      # 물리 연산 시간 간격 (초)
GRAVITY     = 2      # 중력 가속도 (m/s^2)

# ───────── 카메라 상태 ─────────
cam_yaw   = 45.0    # 좌우 회전
cam_pitch = 20.0    # 상하 회전
cam_dist  = 25.0    # 카메라와 타깃 거리
cam_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

left_button_pressed = False
last_mouse_x = 0.0
last_mouse_y = 0.0
is_paused = False

# ────────── 정20면체 데이터 (Unit Icosahedron) ──────────
_phi = (1.0 + math.sqrt(5.0)) / 2.0
_verts_unit = np.array([
    (-1,  _phi,  0),
    ( 1,  _phi,  0),
    (-1, -_phi,  0),
    ( 1, -_phi,  0),
    ( 0, -1,  _phi),
    ( 0,  1,  _phi),
    ( 0, -1, -_phi),
    ( 0,  1, -_phi),
    ( _phi,  0, -1),
    ( _phi,  0,  1),
    (-_phi,  0, -1),
    (-_phi,  0,  1),
], dtype=np.float64)

_faces_idx = [
    (0, 11,  5), (0,  5,  1), (0,  1,  7), (0,  7, 10), (0, 10, 11),
    (1,  5,  9), (5, 11,  4), (11,10,  2), (10, 7,  6), (7,  1,  8),
    (3,  9,  4), (3,  4,  2), (3,  2,  6), (3,  6,  8), (3,  8,  9),
    (4,  9,  5), (2,  4, 11), (6,  2, 10), (8,  6,  7), (9,  8,  1),
]

# 나중에 스케일된 결과를 저장할 전역 변수들
_verts = None           # (12,3) float32
_faces = None           # 훑어볼 면 인덱스 리스트(20개)
_face_normals = None    # (20,3) float32
_face_offsets = None    # (20,)   float32

def _compute_icosahedron_scaled(target_inradius: float):
    """
    1) unit 상태에서 면별 노멀과 offset을 계산 → inradius_unit
    2) scale = target_inradius / inradius_unit
    3) 버텍스 스케일 → 면별 노멀/offset 재계산
    4) 전역 _verts, _faces, _face_normals, _face_offsets 저장
    """
    global _verts, _faces, _face_normals, _face_offsets

    verts_orig = _verts_unit.copy()  # float64

    normals = []
    offsets = []
    for (i0, i1, i2) in _faces_idx:
        p0 = verts_orig[i0]
        p1 = verts_orig[i1]
        p2 = verts_orig[i2]
        n_unn = np.cross(p1 - p0, p2 - p0)
        n_len = np.linalg.norm(n_unn)
        if n_len == 0:
            continue
        n = n_unn / n_len
        d = np.dot(n, p0)
        if d < 0:
            n = -n
            d = -d
        normals.append(n)
        offsets.append(d)

    normals = np.array(normals, dtype=np.float64)
    offsets = np.array(offsets, dtype=np.float64)
    inradius_unit = float(np.mean(offsets))

    # 필요 스케일
    scale = target_inradius / inradius_unit
    verts_scaled = verts_orig * scale  # (12,3) float64

    normals_s = []
    offsets_s = []
    for (i0, i1, i2) in _faces_idx:
        p0 = verts_scaled[i0]
        p1 = verts_scaled[i1]
        p2 = verts_scaled[i2]
        n_unn = np.cross(p1 - p0, p2 - p0)
        n_len = np.linalg.norm(n_unn)
        if n_len == 0:
            continue
        n = n_unn / n_len
        d = np.dot(n, p0)
        if d < 0:
            n = -n
            d = -d
        normals_s.append(n)
        offsets_s.append(d)

    _verts = verts_scaled.astype(np.float32)                        # (12,3)
    _faces = list(_faces_idx)                                        # 면 인덱스
    _face_normals = np.array(normals_s, dtype=np.float32)            # (20,3)
    _face_offsets = np.array(offsets_s, dtype=np.float32)            # (20,)


# ───────── 전역에서 정20면체 스케일 계산 ─────────
CUBE_SIZE = 20.0
_inradius_final = (CUBE_SIZE / 2.0) - RADIUS
_compute_icosahedron_scaled(_inradius_final)


# ───────── Sphere 객체 정의 ─────────
@dataclass
class Sphere:
    pos: np.ndarray      # (3,) float32
    vel: np.ndarray      # (3,) float32
    color: np.ndarray    # (3,) float32 (RGB)
    radius: float = RADIUS
    mass: float = 1.0    # 모두 같은 질량

spheres: List[Sphere] = []
quadric = None         # GLU quadric (Sphere Display List용)

# ───────── 공간 분할(Spatial Grid) 파라미터 ─────────
# 시뮬레이션 공간(정20면체 내부)는 약 inradius_final 크기.
# 이를 작은 정육면체 셀로 분할하여, 충돌 후보를 그리드 단위로 좁힌다.
GRID_CELL_SIZE = RADIUS * 4.0  # 한 셀 가로 길이 (예: 반경*4) → 평균 한 셀당 몇 개 정도의 구가 담김
GRID_DIM = int(math.ceil((2 * _inradius_final) / GRID_CELL_SIZE))
#  그리드 인덱스 범위는 [-GRID_DIM//2 ... +GRID_DIM//2], 3차원

GridKey = Tuple[int, int, int]   # (ix,iy,iz) 정수 인덱스
grid_map: Dict[GridKey, List[int]] = {}

def _grid_key_from_pos(pos: np.ndarray) -> GridKey:
    """
    3D 위치 pos → 그리드 인덱스 (ix, iy, iz)
    0,0,0 셀은 원점 근처. 각 인덱스는 정수.
    """
    half_space = _inradius_final
    rel = (pos + half_space) / GRID_CELL_SIZE  # 0..(2*inradius)/GRID_CELL_SIZE
    ix = int(math.floor(rel[0]))
    iy = int(math.floor(rel[1]))
    iz = int(math.floor(rel[2]))
    return (ix, iy, iz)

def _build_spatial_grid():
    """
    각 구체의 인덱스를 공간 분할 그리드에 등록.
    grid_map: { (ix,iy,iz) : [sphere_index, ...], ... }
    """
    global grid_map
    grid_map.clear()
    for idx, s in enumerate(spheres):
        key = _grid_key_from_pos(s.pos)
        if key not in grid_map:
            grid_map[key] = []
        grid_map[key].append(idx)

def _get_neighbor_candidates(idx: int) -> List[int]:
    """
    sphere[idx] 와 충돌 가능성이 있는 후보 인덱스 목록을 반환.
    해당 구체가 속한 셀과 인접 26개 이웃 셀의 리스트를 합쳐서 검색.
    """
    s = spheres[idx]
    base_key = _grid_key_from_pos(s.pos)
    candidates = []
    bx, by, bz = base_key
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                neighbor_key = (bx + dx, by + dy, bz + dz)
                if neighbor_key in grid_map:
                    candidates.extend(grid_map[neighbor_key])
    # 자기 자신(idx)을 제외
    return [j for j in candidates if j != idx]


# ───────── 구체 초기화 ─────────
def init_spheres():
    """
    100개의 구체를 정20면체 내부의 y ≥ 0 랜덤 위치에 생성.
    서로 겹치지 않도록 검사.
    초기 속도는 모두 0.
    """
    spheres.clear()
    attempts = 0

    while len(spheres) < NUM_SPHERES and attempts < NUM_SPHERES * 200:
        attempts += 1
        x = random.uniform(-_inradius_final, _inradius_final)
        y = random.uniform(0.0, _inradius_final)
        z = random.uniform(-_inradius_final, _inradius_final)
        pt = np.array([x, y, z], dtype=np.float32)

        # 1) 정20면체 내부인지 검사 (각 면 plane: n·pt ≤ offset - radius)
        inside = True
        for i_face in range(len(_face_normals)):
            d = np.dot(_face_normals[i_face], pt)
            if d > _face_offsets[i_face] - RADIUS:
                inside = False
                break
        if not inside:
            continue

        # 2) 기존 구들과 겹치지 않는지 검사
        too_close = False
        for other in spheres:
            if np.linalg.norm(pt - other.pos) < 2 * RADIUS:
                too_close = True
                break
        if too_close:
            continue

        # 3) 최종 확정 → 생성
        vel = np.zeros(3, dtype=np.float32)
        color = np.array([random.random(), random.random(), random.random()], dtype=np.float32)
        spheres.append(Sphere(pos=pt, vel=vel, color=color))

    if len(spheres) < NUM_SPHERES:
        print(f"[경고] 공간 부족으로 {len(spheres)}개만 배치되었습니다.")


# ───────── 물리 업데이트 (최적화 버전) ─────────
def update_physics():
    """
    1) 중력 적용
    2) 위치 갱신
    3) 면(정20면체 삼각형) 충돌 처리
    4) 공간 분할 그리드 빌드
    5) 구-구 충돌 처리 (그리드 후보만 검사)
    """
    if is_paused:
        return

    # 1) 중력 적용 및 위치 갱신
    for s in spheres:
        s.vel[1] -= GRAVITY * DT
        s.pos += s.vel * DT

    # 2) 정20면체 면 충돌
    for s in spheres:
        for i_face in range(len(_face_normals)):
            n = _face_normals[i_face]
            d = np.dot(n, s.pos)
            if d > _face_offsets[i_face] - s.radius:
                penetration = d - (_face_offsets[i_face] - s.radius)
                s.pos -= n * penetration
                v_norm = np.dot(s.vel, n)
                s.vel -= 2 * v_norm * n

    # 3) 공간 분할 그리드 완성
    _build_spatial_grid()

    # 4) 구-구 충돌 (Grid 후보만 검사)
    ncount = len(spheres)
    for i in range(ncount):
        A = spheres[i]
        candidates = _get_neighbor_candidates(i)
        for j in candidates:
            if j <= i:
                continue
            B = spheres[j]
            diff = B.pos - A.pos
            dist = np.linalg.norm(diff)
            min_dist = A.radius + B.radius
            if 0 < dist < min_dist:
                nvec = diff / dist
                rel_vel = np.dot(B.vel - A.vel, nvec)
                if rel_vel < 0:
                    impulse = -2 * rel_vel / (A.mass + B.mass)
                    A.vel -= impulse * nvec * B.mass
                    B.vel += impulse * nvec * A.mass
                    overlap = min_dist - dist
                    A.pos -= nvec * (overlap / 2)
                    B.pos += nvec * (overlap / 2)


# ───────── OpenGL 렌더링 함수 ─────────
def set_camera(window_width, window_height):
    """
    카메라 설정: Perspective + LookAt
    """
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    aspect = window_width / max(window_height, 1)
    gluPerspective(60.0, aspect, 0.1, 100.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    yaw_rad = math.radians(cam_yaw)
    pitch_rad = math.radians(cam_pitch)
    ex = cam_target[0] + cam_dist * math.cos(pitch_rad) * math.sin(yaw_rad)
    ey = cam_target[1] + cam_dist * math.sin(pitch_rad)
    ez = cam_target[2] + cam_dist * math.cos(pitch_rad) * math.cos(yaw_rad)

    gluLookAt(ex, ey, ez,
              cam_target[0], cam_target[1], cam_target[2],
              0.0, 1.0, 0.0)


def draw_icosahedron_wireframe():
    """
    정20면체 와이어프레임 렌더링.
    """
    glColor3f(1.0, 1.0, 1.0)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glBegin(GL_TRIANGLES)
    for (i0, i1, i2) in _faces:
        v0 = _verts[i0]
        v1 = _verts[i1]
        v2 = _verts[i2]
        glVertex3f(v0[0], v0[1], v0[2])
        glVertex3f(v1[0], v1[1], v1[2])
        glVertex3f(v2[0], v2[1], v2[2])
    glEnd()
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)


def draw_spheres():
    """
    Display List로 등록된 sphere_mesh를 each position/color로 그리기.
    """
    global sphere_display_list
    for s in spheres:
        glPushMatrix()
        glTranslatef(s.pos[0], s.pos[1], s.pos[2])
        glColor3f(s.color[0], s.color[1], s.color[2])
        glCallList(sphere_display_list)
        glPopMatrix()


# ───────── GLFW 입력 콜백 ─────────
def key_callback(window, key, scancode, action, mods):
    global is_paused, cam_dist
    if action == glfw.PRESS or action == glfw.REPEAT:
        if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            is_paused = not is_paused
        elif key == glfw.KEY_W:
            cam_dist = max(cam_dist - 0.5, 2.0)
        elif key == glfw.KEY_S:
            cam_dist += 0.5
        elif key == glfw.KEY_A:
            rotate_yaw(-5.0)
        elif key == glfw.KEY_D:
            rotate_yaw(5.0)
        elif key == glfw.KEY_R:
            move_target(0.0, 0.5, 0.0)
        elif key == glfw.KEY_F:
            move_target(0.0, -0.5, 0.0)
        elif key == glfw.KEY_UP:
            rotate_pitch(2.0)
        elif key == glfw.KEY_DOWN:
            rotate_pitch(-2.0)


def mouse_button_callback(window, button, action, mods):
    global left_button_pressed
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            left_button_pressed = True
        elif action == glfw.RELEASE:
            left_button_pressed = False


def cursor_pos_callback(window, xpos, ypos):
    global last_mouse_x, last_mouse_y, cam_yaw, cam_pitch, left_button_pressed
    if left_button_pressed:
        dx = xpos - last_mouse_x
        dy = ypos - last_mouse_y
        # 드래그 방향과 반대로 회전
        cam_yaw   -= dx * 0.3
        cam_pitch -= dy * -0.3
        cam_pitch = max(min(cam_pitch, 89.0), -89.0)
    last_mouse_x = xpos
    last_mouse_y = ypos


def scroll_callback(window, xoffset, yoffset):
    global cam_dist
    if yoffset > 0:
        cam_dist = max(cam_dist * 0.9, 1.0)
    else:
        cam_dist *= 1.1


# ───────── 카메라 제어 헬퍼 ─────────
def rotate_yaw(delta_deg):
    global cam_yaw
    cam_yaw = (cam_yaw + delta_deg) % 360.0


def rotate_pitch(delta_deg):
    global cam_pitch
    cam_pitch = max(min(cam_pitch + delta_deg, 89.0), -89.0)


def move_target(dx=0.0, dy=0.0, dz=0.0):
    global cam_target
    cam_target += np.array([dx, dy, dz], dtype=np.float32)


# ───────── OpenGL 초기화 & Display List 생성 ─────────
sphere_display_list = None

def init_gl():
    """
    OpenGL 설정:
    - 깊이 테스트 켜기
    - 배경색 설정
    - Sphere용 Display List 빌드
    """
    global quadric, sphere_display_list
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.05, 0.05, 0.08, 1.0)

    # 1) GLU Quadric 생성
    quadric = gluNewQuadric()
    gluQuadricNormals(quadric, GLU_SMOOTH)

    # 2) Display List 생성 (구체 메쉬를 미리 컴파일)
    sphere_display_list = glGenLists(1)
    glNewList(sphere_display_list, GL_COMPILE)
    gluSphere(quadric, RADIUS, 16, 16)
    glEndList()


# ───────── 메인 함수 ─────────
def main():
    # 1) GLFW 초기화
    if not glfw.init():
        print("[오류] GLFW 초기화 실패")
        sys.exit(1)

    # OpenGL 2.1 호환성 프로파일
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_ANY_PROFILE)

    # 2) 윈도우 생성
    window_width, window_height = 800, 600
    window = glfw.create_window(window_width, window_height, "Optimized Icosahedron Bouncing", None, None)
    if not window:
        print("[오류] GLFW 창 생성 실패")
        glfw.terminate()
        sys.exit(1)
    glfw.make_context_current(window)

    # 3) 콜백 함수 등록
    glfw.set_key_callback(window, key_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    # 4) 시뮬레이션 초기화
    init_gl()
    init_spheres()

    # 프레임 타이밍: 60 FPS
    prev_time = time.time()
    fps_target = 60.0
    frame_duration = 1.0 / fps_target

    # 5) 메인 루프
    while not glfw.window_should_close(window):
        current_time = time.time()
        elapsed = current_time - prev_time
        if elapsed < frame_duration:
            time.sleep(frame_duration - elapsed)
            current_time = time.time()
            elapsed = current_time - prev_time
        prev_time = current_time

        glfw.poll_events()
        update_physics()

        width, height = glfw.get_framebuffer_size(window)
        glViewport(0, 0, width, height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        set_camera(width, height)
        draw_icosahedron_wireframe()
        draw_spheres()

        glfw.swap_buffers(window)

    # 6) 종료 시 정리
    gluDeleteQuadric(quadric)
    glDeleteLists(sphere_display_list, 1)
    glfw.terminate()
    sys.exit(0)


if __name__ == "__main__":
    main()
