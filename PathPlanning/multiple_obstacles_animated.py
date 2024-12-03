import pygame
import random
import json
import time

with open(r"C:\Users\Gio\farm..tmj", "r") as file:
    map_data = json.load(file)

# Constants
TILE_SIZE = 32
SCREEN_WIDTH = map_data["width"] * TILE_SIZE
SCREEN_HEIGHT = map_data["height"] * TILE_SIZE
tractor_spawn_interval = 1  # seconds between spawns

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Farm Simulation - Tractors")

tile_images = {
    1: pygame.image.load(r"C:\Users\Gio\grass.png").convert_alpha(),
    2: pygame.image.load(r"C:\Users\Gio\weed.png").convert_alpha(),
    3: pygame.image.load(r"C:\Users\Gio\corn.png").convert_alpha(),
    4: pygame.image.load(r"C:\Users\Gio\corn.png").convert_alpha()
}

for key, img in tile_images.items():
    tile_images[key] = pygame.transform.scale(img, (TILE_SIZE, TILE_SIZE))

tractor_images = {
    "left_to_right": pygame.image.load(r"C:\Users\Gio\tractor_left_to_right.png").convert_alpha(),
    "right_to_left": pygame.image.load(r"C:\Users\Gio\tractor_right_to_left.png").convert_alpha(),
    "top_to_bottom": pygame.image.load(r"C:\Users\Gio\tractor_top_to_bottom.png").convert_alpha(),
    "bottom_to_top": pygame.image.load(r"C:\Users\Gio\tractor_bottom_to_top.png").convert_alpha(),
}

for key in tractor_images:
    tractor_images[key] = pygame.transform.scale(tractor_images[key], (TILE_SIZE, TILE_SIZE))

layer_data = map_data["layers"][0]["data"]
map_width = map_data["width"]
map_height = map_data["height"]

grid = [layer_data[i:i + map_width] for i in range(0, len(layer_data), map_width)]

# tractor class
class Tractor:
    def __init__(self, direction, start_pos, tractor_id):
        self.id = tractor_id  # Unique identifier
        self.direction = direction
        self.x, self.y = start_pos
        self.speed = 2  # tiles per second
        self.image = tractor_images[direction]
        self.waiting = False  # for horizontal tractors

    def move(self, tractors):
        # Movement logic remains the same
        if self.check_opposite_collision(tractors):
            return  # Stop movement if a collision is imminent

        if self.waiting:
            self.waiting = any(
                self.check_collision(other)
                for other in tractors
                if other.direction in ["top_to_bottom", "bottom_to_top"]
            )
            if self.waiting:
                return  # Still waiting

        if self.direction == "left_to_right":
            self.x += self.speed / 60.0
        elif self.direction == "right_to_left":
            self.x -= self.speed / 60.0
        elif self.direction == "top_to_bottom":
            self.y += self.speed / 60.0
        elif self.direction == "bottom_to_top":
            self.y -= self.speed / 60.0

        if self.direction in ["left_to_right", "right_to_left"]:
            for other in tractors:
                if other.direction in ["top_to_bottom", "bottom_to_top"]:
                    if self.check_collision(other):
                        self.waiting = True
                        break
            else:
                self.waiting = False

    def draw(self, surface):
        surface.blit(self.image, (int(self.x * TILE_SIZE), int(self.y * TILE_SIZE)))

    def is_off_screen(self):
        return self.x < 0 or self.x >= map_width or self.y < 0 or self.y >= map_height

    def check_collision(self, other):
        self_rect = pygame.Rect(
            int(self.x * TILE_SIZE), int(self.y * TILE_SIZE),
            TILE_SIZE, TILE_SIZE
        )
        other_rect = pygame.Rect(
            int(other.x * TILE_SIZE), int(other.y * TILE_SIZE),
            TILE_SIZE, TILE_SIZE
        )
        return self_rect.colliderect(other_rect)

    def check_opposite_collision(self, tractors):
        for other in tractors:
            if self.direction == "top_to_bottom" and other.direction == "bottom_to_top" and self.x == other.x:
                if abs(self.y - other.y) <= 1:
                    return True
            if self.direction == "bottom_to_top" and other.direction == "top_to_bottom" and self.x == other.x:
                if abs(self.y - other.y) <= 1:
                    return True
            if self.direction == "left_to_right" and other.direction == "right_to_left" and self.y == other.y:
                if abs(self.x - other.x) <= 1:
                    return True
            if self.direction == "right_to_left" and other.direction == "left_to_right" and self.y == other.y:
                if abs(self.x - other.x) <= 1:
                    return True
        return False

tractor_id_counter = 0  # Global counter for tractor IDs

def spawn_random_tractor(tractors):
    global tractor_id_counter

    if len(tractors) >= 20:  # Limit the number of tractors
        return None

    directions = ["left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"]
    direction = random.choice(directions)

    if direction == "left_to_right":
        start_pos = (0, random.randint(0, map_height - 1))
    elif direction == "right_to_left":
        start_pos = (map_width - 1, random.randint(0, map_height - 1))
    elif direction == "top_to_bottom":
        start_pos = (random.randint(0, map_width - 1), 0)
    elif direction == "bottom_to_top":
        start_pos = (random.randint(0, map_width - 1), map_height - 1)

    new_tractor = Tractor(direction, start_pos, tractor_id_counter)
    for tractor in tractors:
        if new_tractor.check_collision(tractor) or new_tractor.check_opposite_collision(tractors):
            return None

    tractor_id_counter += 1  # Increment the ID counter
    return new_tractor
