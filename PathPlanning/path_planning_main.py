import pygame
import json
import random
import math
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import time

with open(r"C:\Users\Gio\farm..tmj", "r") as file:
    map_data = json.load(file)

# constants
TILE_SIZE = 32
SCREEN_WIDTH = map_data["width"] * TILE_SIZE
SCREEN_HEIGHT = map_data["height"] * TILE_SIZE
MAX_ITER = 2500
STEP_SIZE = 25

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("RRT Search and TSP Visualization")
drone_image = pygame.image.load(r"C:\Users\Gio\drone.png").convert_alpha()
drone_image = pygame.transform.scale(drone_image, (TILE_SIZE*2, TILE_SIZE*2))  # scale to match grid tile size


tile_images = {
    1: pygame.image.load(r"C:\Users\Gio\grass.png").convert_alpha(),
    2: pygame.image.load(r"C:\Users\Gio\weed.png").convert_alpha(),
    3: pygame.image.load(r"C:\Users\Gio\corn.png").convert_alpha(),
    4: pygame.image.load(r"C:\Users\Gio\corn.png").convert_alpha()
}

for key, img in tile_images.items():
    tile_images[key] = pygame.transform.scale(img, (TILE_SIZE, TILE_SIZE))

# button settings
BUTTON_WIDTH = 170
BUTTON_HEIGHT = 30
BUTTON_COLOR = (0, 128, 255)
BUTTON_HOVER_COLOR = (0, 102, 204)
BUTTON_TEXT_COLOR = (255, 255, 255)

layer_data = map_data["layers"][0]["data"]
map_width = map_data["width"]
map_height = map_data["height"]
grid = [layer_data[i:i + map_width] for i in range(0, len(layer_data), map_width)]

# button settings (spacing adjusted)
get_targets_button_rect = pygame.Rect(SCREEN_WIDTH - BUTTON_WIDTH - 10, 10, BUTTON_WIDTH, BUTTON_HEIGHT)  # Get Waypoints button
add_button_rect = pygame.Rect(SCREEN_WIDTH - BUTTON_WIDTH - 10, 50, BUTTON_WIDTH, BUTTON_HEIGHT)  # Add Waypoint button
remove_button_rect = pygame.Rect(SCREEN_WIDTH - BUTTON_WIDTH - 10, 90, BUTTON_WIDTH, BUTTON_HEIGHT)  # Remove Waypoint button
run_button_rect = pygame.Rect(SCREEN_WIDTH - BUTTON_WIDTH - 10, 130, BUTTON_WIDTH, BUTTON_HEIGHT)  # Run Waypoint button
button_rect = pygame.Rect(SCREEN_WIDTH - BUTTON_WIDTH - 10, 170, BUTTON_WIDTH, BUTTON_HEIGHT)  # Reroute Path button

# dunction to dynamically update button positions
def update_button_position():
    global button_rect, add_button_rect, remove_button_rect, run_button_rect, get_targets_button_rect
    get_targets_button_rect = pygame.Rect(screen.get_width() - BUTTON_WIDTH - 10, 10, BUTTON_WIDTH, BUTTON_HEIGHT)
    add_button_rect = pygame.Rect(screen.get_width() - BUTTON_WIDTH - 10, 50, BUTTON_WIDTH, BUTTON_HEIGHT)
    remove_button_rect = pygame.Rect(screen.get_width() - BUTTON_WIDTH - 10, 90, BUTTON_WIDTH, BUTTON_HEIGHT)
    run_button_rect = pygame.Rect(screen.get_width() - BUTTON_WIDTH - 10, 130, BUTTON_WIDTH, BUTTON_HEIGHT)
    button_rect = pygame.Rect(screen.get_width() - BUTTON_WIDTH - 10, 170, BUTTON_WIDTH, BUTTON_HEIGHT)

# OR-Tools TSP solver
def tsp_ortools(distance_matrix):
    """Solve TSP using OR-Tools with grid units."""
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    # define cost of each arc
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        cost = int(distance_matrix[from_node][to_node])
        #print(f"Distance callback ({from_node} -> {to_node}): {cost}")
        return cost

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # set arc cost evaluator
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # add Distance Dimension
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        1_000_000,  # arbitrary large max distance
        True,  # start cumul to zero
        "Distance"
    )
    distance_dimension = routing.GetDimensionOrDie("Distance")
    distance_dimension.SetGlobalSpanCostCoefficient(0)  # disable global span cost

    # set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 1

    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        print(f"Raw Objective Value (scaled): {solution.ObjectiveValue()}")
        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(route[0])  # return to start
        total_distance = solution.ObjectiveValue()  # use raw grid units
        return route, total_distance
    else:
        print("No solution found!")
        return [], 0

# calculate distance matrix
def calculate_distance_matrix(weed_locations):
    n = len(weed_locations)
    distance_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = math.dist(weed_locations[i], weed_locations[j])
    # print calculated distance matrix
    print("Distance Matrix (grid units):")
    for row in distance_matrix:
        print(row)
    return distance_matrix

# draw functions
def draw_tsp_path(path, weed_locations, screen):
    font = pygame.font.SysFont(None, 24)

    for i in range(len(path) - 1):
        start = weed_locations[path[i]]
        end = weed_locations[path[i + 1]]
        
        # draw magenta line
        pygame.draw.line(screen, (255, 0, 255), 
                         (start[0] * TILE_SIZE + TILE_SIZE // 2, start[1] * TILE_SIZE + TILE_SIZE // 2),
                         (end[0] * TILE_SIZE + TILE_SIZE // 2, end[1] * TILE_SIZE + TILE_SIZE // 2), 4)
        
        # calculate midpoint of line for placing label
        mid_x = (start[0] * TILE_SIZE + end[0] * TILE_SIZE) // 2 + TILE_SIZE // 2
        mid_y = (start[1] * TILE_SIZE + end[1] * TILE_SIZE) // 2 + TILE_SIZE // 2
        
        # render line number
        label = font.render(str(i + 1), True, (255, 255, 255)) 
        screen.blit(label, (mid_x, mid_y))  # display number at midpoint

start = (0, 0)
# calculate weeds and solve TSP
#found_weeds, rrt_tree = rrt_search(grid, start, MAX_ITER)

# predefined weed locations
found_weeds = [(7, 25), (4, 3), (16, 10), (3, 13), (4, 20), (31, 26), (31, 20), (9, 10), 
               (17, 27), (16, 15), (30, 11), (23, 22), (30, 2)]

if found_weeds:
    print(f"Weed Locations Found: {found_weeds}")
    distance_matrix = calculate_distance_matrix(found_weeds)
    optimal_path, min_distance = tsp_ortools(distance_matrix)

    # ensure path starts from the leftmost point
    if optimal_path:
        leftmost_index = min(range(len(found_weeds)), key=lambda i: found_weeds[i][0])
        leftmost_path_index = optimal_path.index(leftmost_index)
        optimal_path = optimal_path[leftmost_path_index:] + optimal_path[:leftmost_path_index]
        optimal_path.append(optimal_path[0])  # complete the loop

    print(f"Optimal Path (starting from leftmost point): {optimal_path}")
    print(f"Optimal Path Distance (raw grid units): {min_distance:.2f}")
else:
    print("No weeds found!")
    optimal_path, min_distance = [], 0

# drone animation parameters
drone_position = None
drone_speed = 4  # tiles per second
target_index = 0  # current target in the optimal path
movement_progress = 0  # progress between two points (0.0 to 1.0)
drone_active = False  # Controls whether the drone is active

# spawn the drone at the leftmost target
if found_weeds and optimal_path:
    leftmost_index = min(range(len(found_weeds)), key=lambda i: found_weeds[optimal_path[i]][0])
    drone_position = found_weeds[optimal_path[leftmost_index]]
    target_index = leftmost_index

# function to move the drone
def move_drone(delta_time):
    global drone_position, target_index, movement_progress, drone_active

    if drone_active and target_index < len(optimal_path) - 1:
        current_target = found_weeds[optimal_path[target_index]]
        next_target = found_weeds[optimal_path[target_index + 1]]

        # calculate movement direction and progress
        direction = (
            next_target[0] - current_target[0],
            next_target[1] - current_target[1]
        )
        distance = math.dist(current_target, next_target)

        if distance == 0:  # handle zero distance case
            print(f"Zero distance detected between {current_target} and {next_target}. Skipping to next target.")
            movement_progress = 0.0
            target_index += 1
            return

        # update progress based on speed
        movement_progress += drone_speed * delta_time / distance

        if movement_progress >= 1.0:
            # move to the next target
            movement_progress = 0.0
            target_index += 1
            drone_position = next_target
            print(f"Drone reached: {drone_position}")
        else:
            # interpolate position
            drone_position = (
                current_target[0] + movement_progress * direction[0],
                current_target[1] + movement_progress * direction[1]
            )

running = True
font = pygame.font.SysFont(None, 24)
clock = pygame.time.Clock()
last_spawn_time = time.time()

# add flags to control the display and calculation logic
display_weeds = False  # initially, weeds are not displayed
add_waypoint_mode = False  # initially, the mode is inactive
remove_waypoint_mode = False  # initially, the mode is inactive
calculate_tsp_path = False  # initially, TSP path is not calculated or displayed

# main loop
while running:
    delta_time = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.VIDEORESIZE:
            SCREEN_WIDTH, SCREEN_HEIGHT = event.w, event.h
            screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
            update_button_position()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # add Waypoint button click
            if add_button_rect.collidepoint(event.pos):
                add_waypoint_mode = not add_waypoint_mode  # toggle mode
                print("Add Waypoint mode:", "Active" if add_waypoint_mode else "Inactive")
            # add waypoint on screen click
            elif add_waypoint_mode:
                x, y = event.pos
                grid_x, grid_y = x // TILE_SIZE, y // TILE_SIZE  # convert pixel to grid coordinates
                if 0 <= grid_x < map_width and 0 <= grid_y < map_height:
                    new_waypoint = (grid_x, grid_y)
                    found_weeds.append(new_waypoint)
                    print(f"New waypoint added: {new_waypoint}")
            # remove Waypoint button click
            elif remove_button_rect.collidepoint(event.pos):
                remove_waypoint_mode = not remove_waypoint_mode  # toggle mode
                print("Remove Waypoint mode:", "Active" if remove_waypoint_mode else "Inactive")
            # remove waypoint on screen click
            elif remove_waypoint_mode:
                x, y = event.pos
                grid_x, grid_y = x // TILE_SIZE, y // TILE_SIZE  # convert pixel to grid coordinates
                if (grid_x, grid_y) in found_weeds:
                    found_weeds.remove((grid_x, grid_y))
                    print(f"Waypoint removed: {(grid_x, grid_y)}")
            # run Optimal Path button click
            elif run_button_rect.collidepoint(event.pos):
                print("Run Optimal Path button clicked!")
                if display_weeds:  # Ensure weeds are displayed before running TSP
                    print("Calculating TSP path...")
                    distance_matrix = calculate_distance_matrix(found_weeds)
                    optimal_path, min_distance = tsp_ortools(distance_matrix)
                    calculate_tsp_path = True  # Enable TSP path display
            # Get Targets button click
            elif get_targets_button_rect.collidepoint(event.pos):
                print("Get Targets button clicked!")
                display_weeds = True  # display the weeds
                calculate_tsp_path = False  # do not calculate or display TSP path yet
            # execute Optimal Path button click
            elif button_rect.collidepoint(event.pos):
                print("Execute Optimal Path button clicked!")
                if calculate_tsp_path:
                    print("Starting drone navigation...")
                    target_index = 0
                    drone_position = found_weeds[optimal_path[target_index]]  # set drone starting position
                    drone_active = True  # activate the drone

    # clear the screen
    screen.fill((0, 0, 0))

    # draw grid tiles
    for y in range(map_height):
        for x in range(map_width):
            tile_value = grid[y][x]
            if tile_value in tile_images:
                screen.blit(tile_images[tile_value], (x * TILE_SIZE, y * TILE_SIZE))

    # conditionally draw weed locations
    if display_weeds:
        for (x, y) in found_weeds:
            pygame.draw.circle(screen, (255, 0, 0), (x * TILE_SIZE + TILE_SIZE // 2, y * TILE_SIZE + TILE_SIZE // 2), TILE_SIZE // 3)

    # conditionally draw TSP path
    if calculate_tsp_path and display_weeds and found_weeds:
        draw_tsp_path(optimal_path, found_weeds, screen)

    # move and draw the drone only if active
    if drone_active and drone_position:
        move_drone(delta_time)
        drone_pixel_position = (
            int(drone_position[0] * TILE_SIZE),
            int(drone_position[1] * TILE_SIZE)
        )
        screen.blit(drone_image, drone_pixel_position)

    # draw buttons
    mouse_pos = pygame.mouse.get_pos()

    # get Targets button
    get_targets_button_color = (220, 165, 0) if get_targets_button_rect.collidepoint(mouse_pos) else (204, 153, 0)
    pygame.draw.rect(screen, get_targets_button_color, get_targets_button_rect)
    get_targets_button_text = font.render("Get Targets", True, BUTTON_TEXT_COLOR)
    screen.blit(get_targets_button_text, get_targets_button_text.get_rect(center=get_targets_button_rect.center))

    # add Waypoint button
    add_button_color = (0, 255, 0) if add_waypoint_mode else (0, 150, 0)
    pygame.draw.rect(screen, add_button_color, add_button_rect)
    add_button_text = font.render("Add Waypoint", True, BUTTON_TEXT_COLOR)
    screen.blit(add_button_text, add_button_text.get_rect(center=add_button_rect.center))

    # remove Waypoint button
    remove_button_color = (255, 0, 0) if remove_waypoint_mode else (150, 0, 0)
    pygame.draw.rect(screen, remove_button_color, remove_button_rect)
    remove_button_text = font.render("Remove Waypoint", True, BUTTON_TEXT_COLOR)
    screen.blit(remove_button_text, remove_button_text.get_rect(center=remove_button_rect.center))

    # run TSP button
    run_button_color = (0, 0, 180) if run_button_rect.collidepoint(mouse_pos) else (0, 0, 128)
    pygame.draw.rect(screen, run_button_color, run_button_rect)
    run_button_text = font.render("Get Optimal Path", True, BUTTON_TEXT_COLOR)
    screen.blit(run_button_text, run_button_text.get_rect(center=run_button_rect.center))

    # execute Optimal Path button
    button_color = BUTTON_HOVER_COLOR if button_rect.collidepoint(mouse_pos) else BUTTON_COLOR
    pygame.draw.rect(screen, button_color, button_rect)
    button_text = font.render("Execute Optimal Plan", True, BUTTON_TEXT_COLOR)
    screen.blit(button_text, button_text.get_rect(center=button_rect.center))

    # display total distance
    if calculate_tsp_path:
        distance_text = f"Optimal Path Distance: {min_distance:.2f} units"
        screen.blit(font.render(distance_text, True, (255, 255, 255)), (10, 10))

    pygame.display.flip()

pygame.quit()