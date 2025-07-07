import pygame
import sys
import math
import numpy as np

# Constants
WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (10, 10, 50)
LINE_COLOR = (200, 200, 200)
FPS = 60

class Camera:
    def __init__(self, position=None, rotation=None, focal_length=500):
        self.position = np.array(position if position else [0, 0, -5], dtype=float)
        # rotation: pitch (x-axis), yaw (y-axis), roll (z-axis)
        self.rotation = np.array(rotation if rotation else [0, 0, 0], dtype=float)
        self.focal_length = focal_length

    def project(self, vertex):
        # Transform to camera space
        x, y, z = vertex - self.position
        pitch, yaw, roll = self.rotation
        # Pitch (X-axis)
        cosb, sinb = math.cos(pitch), math.sin(pitch)
        y, z = y*cosb - z*sinb, y*sinb + z*cosb
        # Yaw (Y-axis)
        cosa, sina = math.cos(yaw), math.sin(yaw)
        x, z = x*cosa - z*sina, x*sina + z*cosa
        # Roll (Z-axis)
        cosc, sinc = math.cos(roll), math.sin(roll)
        x, y = x*cosc - y*sinc, x*sinc + y*cosc
        # Perspective projection
        if z == 0:
            z = 1e-5
        factor = self.focal_length / z
        xp = x * factor + WIDTH / 2
        yp = -y * factor + HEIGHT / 2
        return (int(xp), int(yp))

class Mesh:
    def __init__(self, vertices, edges):
        self.vertices = np.array(vertices, dtype=float)
        self.edges = edges

    @staticmethod
    def cube(size=2):
        s = size / 2
        verts = [
            [-s, -s, -s], [ s, -s, -s], [ s,  s, -s], [-s,  s, -s],
            [-s, -s,  s], [ s, -s,  s], [ s,  s,  s], [-s,  s,  s]
        ]
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        return Mesh(verts, edges)

    @staticmethod
    def tetrahedron(size=2):
        s = size / math.sqrt(2)
        verts = [
            [ s, 0, -s/math.sqrt(2)], [-s, 0, -s/math.sqrt(2)],
            [0,  s,  s/math.sqrt(2)], [0, -s,  s/math.sqrt(2)]
        ]
        edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        return Mesh(verts, edges)

    @staticmethod
    def sphere(radius=1, stacks=12, slices=24):
        verts = []
        edges = []
        # generate vertices
        for i in range(stacks+1):
            phi = math.pi * i / stacks
            for j in range(slices):
                theta = 2 * math.pi * j / slices
                x = radius * math.sin(phi) * math.cos(theta)
                y = radius * math.cos(phi)
                z = radius * math.sin(phi) * math.sin(theta)
                verts.append([x, y, z])
        # connect edges
        def idx(i, j): return i * slices + j
        for i in range(stacks+1):
            for j in range(slices):
                edges.append((idx(i,j), idx(i, (j+1)%slices)))
                if i < stacks:
                    edges.append((idx(i,j), idx(i+1, j)))
        return Mesh(verts, edges)

    def load_stl(self, filename):
        # TODO: implement STL loading
        pass

class Renderer:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("3D Renderer")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.camera = Camera()
        self.meshes = {
            'cube': Mesh.cube(),
            'tetra': Mesh.tetrahedron(),
            'sphere': Mesh.sphere()
        }
        self.current = 'cube'
        self.drag = False
        self.button = None
        self.last_pos = None

    def run(self):
        while True:
            self._handle_events()
            self._draw()
            pygame.display.flip()
            self.clock.tick(FPS)

    def _handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_1:
                    self.current = 'cube'
                elif ev.key == pygame.K_2:
                    self.current = 'tetra'
                elif ev.key == pygame.K_3:
                    self.current = 'sphere'
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                self.drag = True
                self.button = ev.button
                self.last_pos = np.array(pygame.mouse.get_pos(), dtype=float)
                if ev.button == 4:
                    self.camera.focal_length *= 1.1
                elif ev.button == 5:
                    self.camera.focal_length /= 1.1
            elif ev.type == pygame.MOUSEBUTTONUP:
                self.drag = False
            elif ev.type == pygame.MOUSEMOTION and self.drag:
                curr = np.array(ev.pos, dtype=float)
                diff = curr - self.last_pos
                self.last_pos = curr
                if self.button == 1:
                    # 좌클릭: 카메라 평행 이동
                    self.camera.position += np.array([-diff[0], diff[1], 0]) * 0.01
                elif self.button == 3:
                    # 우클릭: 카메라 회전
                    self.camera.rotation[1] += diff[0] * 0.005  # yaw
                    self.camera.rotation[0] += diff[1] * 0.005  # pitch

    def _draw(self):
        self.screen.fill(BACKGROUND_COLOR)
        mesh = self.meshes[self.current]
        for e in mesh.edges:
            p1 = self.camera.project(mesh.vertices[e[0]])
            p2 = self.camera.project(mesh.vertices[e[1]])
            pygame.draw.line(self.screen, LINE_COLOR, p1, p2)

if __name__ == '__main__':
    Renderer().run()
