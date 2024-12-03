import pygame
import json
import random
import math
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import time
from multiple_obstacles_animated import Tractor, spawn_random_tractor
from collections import deque

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
tsp_time = 0.0
tractor_spawn_interval = 2  # seconds between spawns

# button settings (spacing adjusted)
get_targets_button_rect = pygame.Rect(SCREEN_WIDTH - BUTTON_WIDTH - 10, 10, BUTTON_WIDTH, BUTTON_HEIGHT)  # Get Waypoints button
add_button_rect = pygame.Rect(SCREEN_WIDTH - BUTTON_WIDTH - 10, 50, BUTTON_WIDTH, BUTTON_HEIGHT)  # Add Waypoint button
remove_button_rect = pygame.Rect(SCREEN_WIDTH - BUTTON_WIDTH - 10, 90, BUTTON_WIDTH, BUTTON_HEIGHT)  # Remove Waypoint button
run_button_rect = pygame.Rect(SCREEN_WIDTH - BUTTON_WIDTH - 10, 130, BUTTON_WIDTH, BUTTON_HEIGHT)  # Run Waypoint button
button_rect = pygame.Rect(SCREEN_WIDTH - BUTTON_WIDTH - 10, 170, BUTTON_WIDTH, BUTTON_HEIGHT)  # Reroute Path button
a_star_rect = pygame.Rect(SCREEN_WIDTH - BUTTON_WIDTH - 10, 210, BUTTON_WIDTH, BUTTON_HEIGHT)  # astar button

# dunction to dynamically update button positions
def update_button_position():
    global button_rect, add_button_rect, remove_button_rect, run_button_rect, get_targets_button_rect, a_star_rect
    get_targets_button_rect = pygame.Rect(screen.get_width() - BUTTON_WIDTH - 10, 10, BUTTON_WIDTH, BUTTON_HEIGHT)
    add_button_rect = pygame.Rect(screen.get_width() - BUTTON_WIDTH - 10, 50, BUTTON_WIDTH, BUTTON_HEIGHT)
    remove_button_rect = pygame.Rect(screen.get_width() - BUTTON_WIDTH - 10, 90, BUTTON_WIDTH, BUTTON_HEIGHT)
    run_button_rect = pygame.Rect(screen.get_width() - BUTTON_WIDTH - 10, 130, BUTTON_WIDTH, BUTTON_HEIGHT)
    button_rect = pygame.Rect(screen.get_width() - BUTTON_WIDTH - 10, 170, BUTTON_WIDTH, BUTTON_HEIGHT)
    a_star_rect = pygame.Rect(screen.get_width() - BUTTON_WIDTH - 10, 210, BUTTON_WIDTH, BUTTON_HEIGHT)

# OR-Tools TSP solver
def tsp_ortools(distance_matrix):
    """Solve TSP using OR-Tools with grid units and measure execution time."""
    start_time = time.time()  # Start timing

    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Define cost of each arc
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        cost = int(distance_matrix[from_node][to_node])
        return cost

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Set arc cost evaluator
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance Dimension
    routing.AddDimension(
        transit_callback_index,
        0,  # No slack
        1_000_000,  # Arbitrary large max distance
        True,  # Start cumul to zero
        "Distance"
    )
    distance_dimension = routing.GetDimensionOrDie("Distance")
    distance_dimension.SetGlobalSpanCostCoefficient(0)  # Disable global span cost

    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 1

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)
    end_time = time.time()  # End timing

    execution_time = end_time - start_time  # Calculate elapsed time
    print(f"TSP calculation took {execution_time:.4f} seconds")

    if solution:
        print(f"Raw Objective Value (scaled): {solution.ObjectiveValue()}")
        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(route[0])  # Return to start
        total_distance = solution.ObjectiveValue()  # Use raw grid units
        return route, total_distance, execution_time
    else:
        print("No solution found!")
        return [], 0, 0

# calculate distance matrix
def calculate_distance_matrix(weed_locations):
    # Extract only the coordinates (x, y) from weed_locations
    coordinates = [location[0] for location in weed_locations]
    n = len(coordinates)
    distance_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = math.dist(coordinates[i], coordinates[j])
    # Print calculated distance matrix
    print("Distance Matrix (grid units):")
    for row in distance_matrix:
        print(row)
    return distance_matrix


# draw functions
def draw_tsp_path(path, weed_locations, screen):
    font = pygame.font.SysFont(None, 24)

    for i in range(len(path) - 1):
        start = weed_locations[path[i]][0]  # Extract (x, y) coordinates
        end = weed_locations[path[i + 1]][0]  # Extract (x, y) coordinates
        
        # Draw magenta line
        pygame.draw.line(screen, (255, 0, 255), 
                         (start[0] * TILE_SIZE + TILE_SIZE // 2, start[1] * TILE_SIZE + TILE_SIZE // 2),
                         (end[0] * TILE_SIZE + TILE_SIZE // 2, end[1] * TILE_SIZE + TILE_SIZE // 2), 4)
        
        # Calculate midpoint of the line for placing label
        mid_x = (start[0] * TILE_SIZE + end[0] * TILE_SIZE) // 2 + TILE_SIZE // 2
        mid_y = (start[1] * TILE_SIZE + end[1] * TILE_SIZE) // 2 + TILE_SIZE // 2
        
        # Render line number
        label = font.render(str(i + 1), True, (255, 255, 255)) 
        screen.blit(label, (mid_x, mid_y))  # Display number at midpoint


start = (0, 0)
# calculate weeds and solve TSP
#found_weeds, rrt_tree = rrt_search(grid, start, MAX_ITER)

# predefined weed locations
found_weeds = [
    ((7, 25), 'red'), ((4, 3), 'red'), ((16, 10), 'red'), ((3, 13), 'red'),
    ((4, 20), 'red'), ((31, 26), 'red'), ((31, 20), 'red'), ((9, 10), 'red'),
    ((17, 27), 'red'), ((16, 15), 'red'), ((30, 11), 'red'), ((23, 22), 'red'),
    ((30, 2), 'red')
]

if found_weeds:
    print(f"Weed Locations Found: {found_weeds}")
    distance_matrix = calculate_distance_matrix(found_weeds)
    optimal_path, min_distance, tsp_time = tsp_ortools(distance_matrix)

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

# Spawn the drone at the leftmost target
if found_weeds and optimal_path:
    leftmost_index = min(range(len(found_weeds)), key=lambda i: found_weeds[optimal_path[i]][0][0])
    drone_position = found_weeds[optimal_path[leftmost_index]][0]  # Extract (x, y) coordinates
    target_index = leftmost_index

def is_collision_with_tractor(drone_position, tractors):
    detection_range = 2 * TILE_SIZE  # Set range to 2 tiles
    drone_rect = pygame.Rect(
        drone_position[0] * TILE_SIZE - detection_range // 2,
        drone_position[1] * TILE_SIZE - detection_range // 2,
        TILE_SIZE + detection_range,
        TILE_SIZE + detection_range
    )
    for tractor in tractors:
        tractor_rect = pygame.Rect(
            tractor.x * TILE_SIZE, tractor.y * TILE_SIZE, 
            TILE_SIZE, TILE_SIZE
        )
        if drone_rect.colliderect(tractor_rect):
            return True
    return False

# Add a global timer and waiting flag
drone_waiting = False
wait_start_time = 0

# Add new global variables for recalculated paths
drone_recalculated_path = []
recalculating_path = False

def a_star_pathfinding(start, goal, tractors):
    queue = deque([start])
    visited = set()
    came_from = {}

    while queue:
        current = queue.popleft()
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        visited.add(current)
        x, y = current

        # Explore neighbors (Manhattan movement)
        neighbors = [
            (x + 1, y), (x - 1, y),  # Horizontal
            (x, y + 1), (x, y - 1),  # Vertical
        ]
        for nx, ny in neighbors:
            if 0 <= nx < map_width and 0 <= ny < map_height and (nx, ny) not in visited:
                if not any(
                    abs(int(tractor.x) - nx) == 0 and
                    abs(int(tractor.y) - ny) == 0
                    for tractor in tractors
                ):
                    queue.append((nx, ny))
                    came_from[(nx, ny)] = current

    return []  # Return empty path if no solution is found

def move_drone(delta_time, tractors):
    global drone_position, target_index, movement_progress, drone_active
    global drone_waiting, wait_start_time, drone_recalculated_path, recalculating_path
    global recalculated_progress, recalculated_target_index

    if not drone_active or target_index >= len(optimal_path) - 1:
        return

    # If waiting after sidestepping
    if drone_waiting:
        if time.time() - wait_start_time >= 1:
            drone_waiting = False
            print("Drone resuming movement.")
        else:
            return

    # If following a recalculated path
    if recalculating_path:
        if recalculated_target_index < len(drone_recalculated_path) - 1:
            current_target = drone_recalculated_path[recalculated_target_index]
            next_target = drone_recalculated_path[recalculated_target_index + 1]
            
            # Calculate direction and distance
            direction = (
                next_target[0] - current_target[0],
                next_target[1] - current_target[1]
            )
            distance = math.dist(current_target, next_target)

            # Update the movement progress
            recalculated_progress += drone_speed * delta_time / distance

            if recalculated_progress >= 1.0:
                # Move to the next target in the recalculated path
                recalculated_progress = 0.0
                recalculated_target_index += 1
                drone_position = next_target
                print(f"Drone progressing on recalculated path: {drone_position}")
            else:
                # Interpolate position
                drone_position = (
                    current_target[0] + recalculated_progress * direction[0],
                    current_target[1] + recalculated_progress * direction[1]
                )
        else:
            # Finished recalculated path; rejoin TSP path
            recalculating_path = False
            recalculated_target_index = 0
            recalculated_progress = 0.0
            print(f"Drone rejoined TSP path at {drone_position}")
        return

    # Default movement along TSP path

    current_target = found_weeds[optimal_path[target_index]][0]  # Extract (x, y)
    next_target = found_weeds[optimal_path[target_index + 1]][0]  # Extract (x, y)
# Convert `drone_position` to grid coordinates
    drone_grid_position = (int(drone_position[0]), int(drone_position[1]))

    # Check for tractors nearby
    for tractor in tractors:
        tractor_position = (tractor.x, tractor.y)
        drone_grid_position = (int(drone_position[0]), int(drone_position[1]))
        if abs(drone_grid_position[0] - int(tractor_position[0])) <= 1 and \
           abs(drone_grid_position[1] - int(tractor_position[1])) <= 1:
            # Sidestep and start recalculating
            print(f"Drone avoiding tractor at {tractor_position}")
            x, y = drone_grid_position
            sidestep_positions = [
                (x + 1, y), (x - 1, y),
                (x, y + 1), (x, y - 1),
            ]
            for new_x, new_y in sidestep_positions:
                if 0 <= new_x < map_width and 0 <= new_y < map_height:
                    if not any(
                        abs(int(tractor.x) - new_x) == 0 and
                        abs(int(tractor.y) - new_y) == 0
                        for tractor in tractors
                    ):
                        drone_position = (new_x, new_y)
                        drone_waiting = True
                        wait_start_time = time.time()
                        # Recalculate path to next target
                        drone_recalculated_path = a_star_pathfinding(drone_position, next_target, tractors)
                        recalculating_path = True if drone_recalculated_path else False
                        recalculated_target_index = 0
                        recalculated_progress = 0.0
                        print(f"Recalculated path: {drone_recalculated_path}")
                        return

    # Default TSP movement logic
    direction = (
        next_target[0] - current_target[0],
        next_target[1] - current_target[1],
    )
    distance = math.dist(current_target, next_target)

    if distance == 0:  # Skip if already at the target
        movement_progress = 0.0
        target_index += 1
        return

    # Update the movement progress toward the next TSP waypoint
    movement_progress += drone_speed * delta_time / distance

    if movement_progress >= 1.0:
        movement_progress = 0.0
        target_index += 1
        drone_position = next_target
        print(f"Drone reached: {drone_position}")

        # Update weed status to green
        for i, ((wx, wy), color) in enumerate(found_weeds):
            if (wx, wy) == next_target:
                found_weeds[i] = ((wx, wy), 'green')
                break
    else:
        # Interpolate position toward the next target
        drone_position = (
            current_target[0] + movement_progress * direction[0],
            current_target[1] + movement_progress * direction[1],
        )



# Main loop
running = True
font = pygame.font.SysFont(None, 24)
clock = pygame.time.Clock()
last_spawn_time = time.time()

# Flags to control the display and calculation logic
display_weeds = False  # Initially, weeds are not displayed
add_waypoint_mode = False  # Initially, the mode is inactive
remove_waypoint_mode = False  # Initially, the mode is inactive
calculate_tsp_path = False  # Initially, TSP path is not calculated or displayed
tsp_time_display = 0.0  # Store the TSP calculation time for display
tractors = []  # List to hold active tractors

# Main loop
# Main loop
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
            # Add Waypoint button click
            if add_button_rect.collidepoint(event.pos):
                add_waypoint_mode = not add_waypoint_mode  # Toggle mode
                print("Add Waypoint mode:", "Active" if add_waypoint_mode else "Inactive")
            # Add waypoint on screen click
            elif add_waypoint_mode:
                x, y = event.pos
                grid_x, grid_y = x // TILE_SIZE, y // TILE_SIZE  # Convert pixel to grid coordinates
                if 0 <= grid_x < map_width and 0 <= grid_y < map_height:
                    new_waypoint = ((grid_x, grid_y), 'red')  # Ensure correct format
                    found_weeds.append(new_waypoint)
                    print(f"New waypoint added: {new_waypoint}")

            # Remove Waypoint button click
            elif remove_button_rect.collidepoint(event.pos):
                remove_waypoint_mode = not remove_waypoint_mode  # Toggle mode
                print("Remove Waypoint mode:", "Active" if remove_waypoint_mode else "Inactive")
            # Remove waypoint on screen click
            elif remove_waypoint_mode:
                x, y = event.pos
                grid_x, grid_y = x // TILE_SIZE, y // TILE_SIZE  # Convert pixel to grid coordinates
                for waypoint in found_weeds:
                    if waypoint[0] == (grid_x, grid_y):  # Compare coordinates only
                        found_weeds.remove(waypoint)
                        print(f"Waypoint removed: {(grid_x, grid_y)}")
                        break  # Exit after removing to avoid modifying the list during iteration

            # Run Optimal Path button click
            elif run_button_rect.collidepoint(event.pos):
                print("Run Optimal Path button clicked!")
                if display_weeds:  # Ensure weeds are displayed before running TSP
                    print("Calculating TSP path...")
                    distance_matrix = calculate_distance_matrix(found_weeds)
                    optimal_path, min_distance, tsp_time = tsp_ortools(distance_matrix)  # Correct unpacking
                    tsp_time_display = tsp_time  # Store TSP time for display
                    calculate_tsp_path = True  # Enable TSP path display
                    print(f"TSP Calculation Time: {tsp_time:.4f} seconds")  # For debugging

            # Get Targets button click
            elif get_targets_button_rect.collidepoint(event.pos):
                print("Get Targets button clicked!")
                display_weeds = True  # Display the weeds
                calculate_tsp_path = False  # Do not calculate or display TSP path yet
            # Execute Optimal Path button click
            elif button_rect.collidepoint(event.pos):
                print("Execute Optimal Path button clicked!")
                if calculate_tsp_path:
                    drone_active = True  # Activate the drone
            # A* Button click
            elif a_star_rect.collidepoint(event.pos):
                print("A* button clicked!")
                # Placeholder logic for enabling A* mode or triggering A* pathfinding
                a_star_enabled = not globals().get("a_star_enabled", False)  # Toggle A* mode
                print(f"A* Algorithm mode: {'Enabled' if a_star_enabled else 'Disabled'}")

    # Spawn a new tractor periodically if less than 6 are active
    if time.time() - last_spawn_time > tractor_spawn_interval and len(tractors) < 6:
        new_tractor = spawn_random_tractor(tractors)
        if new_tractor:
            tractors.append(new_tractor)
            last_spawn_time = time.time()

    # Move and update tractors
    for tractor in tractors[:]:
        tractor.move(tractors)
        #print(f"Tractor {tractor.id} at ({tractor.x:.2f}, {tractor.y:.2f}) moving {tractor.direction}")  # Log position
        if tractor.is_off_screen():
            tractors.remove(tractor)

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw grid tiles
    for y in range(map_height):
        for x in range(map_width):
            tile_value = grid[y][x]
            if tile_value in tile_images:
                screen.blit(tile_images[tile_value], (x * TILE_SIZE, y * TILE_SIZE))

    # Conditionally draw weed locations
# Conditionally draw weed locations
    if display_weeds:
        for ((x, y), color) in found_weeds:
            weed_color = (0, 255, 0) if color == 'green' else (255, 0, 0)
            pygame.draw.circle(screen, weed_color, 
                            (x * TILE_SIZE + TILE_SIZE // 2, y * TILE_SIZE + TILE_SIZE // 2), 
                            TILE_SIZE // 3)

    # Conditionally draw TSP path
    if calculate_tsp_path and display_weeds and found_weeds:
        draw_tsp_path(optimal_path, found_weeds, screen)

    # Move and draw the drone only if active
    if drone_active and drone_position:
        move_drone(delta_time, tractors)  # Pass both parameters: delta_time and tractors
        drone_pixel_position = (
            int(drone_position[0] * TILE_SIZE),
            int(drone_position[1] * TILE_SIZE)
        )
        screen.blit(drone_image, drone_pixel_position)


    # Draw tractors
    for tractor in tractors:
        tractor.draw(screen)

    # Draw buttons
    mouse_pos = pygame.mouse.get_pos()

    # Get Targets button
    get_targets_button_color = (220, 165, 0) if get_targets_button_rect.collidepoint(mouse_pos) else (204, 153, 0)
    pygame.draw.rect(screen, get_targets_button_color, get_targets_button_rect)
    get_targets_button_text = font.render("Get Targets", True, BUTTON_TEXT_COLOR)
    screen.blit(get_targets_button_text, get_targets_button_text.get_rect(center=get_targets_button_rect.center))

    # Add Waypoint button
    add_button_color = (0, 255, 0) if add_waypoint_mode else (0, 150, 0)
    pygame.draw.rect(screen, add_button_color, add_button_rect)
    add_button_text = font.render("Add Waypoint", True, BUTTON_TEXT_COLOR)
    screen.blit(add_button_text, add_button_text.get_rect(center=add_button_rect.center))

    # Remove Waypoint button
    remove_button_color = (255, 0, 0) if remove_waypoint_mode else (150, 0, 0)
    pygame.draw.rect(screen, remove_button_color, remove_button_rect)
    remove_button_text = font.render("Remove Waypoint", True, BUTTON_TEXT_COLOR)
    screen.blit(remove_button_text, remove_button_text.get_rect(center=remove_button_rect.center))

    # Run TSP button
    run_button_color = (0, 0, 180) if run_button_rect.collidepoint(mouse_pos) else (0, 0, 128)
    pygame.draw.rect(screen, run_button_color, run_button_rect)
    run_button_text = font.render("Get Optimal Path", True, BUTTON_TEXT_COLOR)
    screen.blit(run_button_text, run_button_text.get_rect(center=run_button_rect.center))

    # Execute Optimal Path button
    button_color = BUTTON_HOVER_COLOR if button_rect.collidepoint(mouse_pos) else BUTTON_COLOR
    pygame.draw.rect(screen, button_color, button_rect)
    button_text = font.render("Execute Optimal Plan", True, BUTTON_TEXT_COLOR)
    screen.blit(button_text, button_text.get_rect(center=button_rect.center))

    # Execute A* button
    astarbutton_color = (255, 165, 0) if a_star_rect.collidepoint(mouse_pos) else (255, 140, 0)  # Orange shades
    pygame.draw.rect(screen, astarbutton_color, a_star_rect)
    astarbutton_text = font.render("Enable A*", True, BUTTON_TEXT_COLOR)
    screen.blit(astarbutton_text, astarbutton_text.get_rect(center=a_star_rect.center))

    # Display TSP calculation time and distance
    if calculate_tsp_path:
        distance_text = f"Optimal Path Distance: {min_distance:.2f} units"
        tsp_time_text = f"TSP Calculation Time: {tsp_time_display:.4f} seconds"
        screen.blit(font.render(distance_text, True, (255, 255, 255)), (10, 10))
        screen.blit(font.render(tsp_time_text, True, (255, 255, 255)), (10, SCREEN_HEIGHT - 40))  # Adjusted for bottom

    pygame.display.flip()

pygame.quit()

