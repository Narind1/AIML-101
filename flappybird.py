import pygame
import random

pygame.init()

black=(0,0,0)
blue=(135,206,235)
green=(00,200,00)
orange=(250,175,0)

# Dimensons of screen
s_width = 600
s_height = 600

# Attributes of bird
bird_size = 30
bird_x = 50
bird_y = s_height // 2
gravity = 0.25
bird_jump = -4.5 

# Attributes of pipe
pipe_width = 50
pipe_height = random.randint(200, 400)
pipe_x = s_width
pipe_speed = 6
pip_gap = 150

# Set up for screen
scrn=pygame.display.set_mode((s_width,s_height))
pygame.display.set_caption("FLAPPY_BIRD")

# Initialising the score
scores= []

# FPS
clock = pygame.time.Clock()

# Functions for all attributes
def draw_bird(x, y):
    pygame.draw.rect(scrn, orange , (x, y, bird_size,bird_size))

def draw_pipe(x, height):
    pygame.draw.rect(scrn,green,(x,0,pipe_width,height))
    pygame.draw.rect(scrn,green,(x,height+pip_gap,pipe_width,s_height-height-pip_gap))

def reset_pipe():
    global pipe_height, pipe_x
    pipe_height = random.randint(200, 400)
    pipe_x = s_width

def display_score(score):
    font = pygame.font.SysFont( None , 50)
    text = font.render(" Score : "+ str(score) , True , black)
    scrn.blit(text,(10,10))

def check_collision():
    if bird_x + bird_size > pipe_x and bird_x < pipe_x + pipe_width:
        if bird_y < pipe_height or bird_y + bird_size > pipe_height + pip_gap:
            return True
    return False

def final_score(score):
    font=pygame.font.SysFont(None,70)
    text=font.render("Final Score: "+ str(score), True, black)
    text_rect=text.get_rect(center=(s_width//2,s_height//2))
    scrn.blit(text,text_rect)

    font=pygame.font.SysFont(None,30)
    restart_text = font.render("Click to restart", True, black)
    restart_text_rect = restart_text.get_rect(center=(s_width // 2, s_height // 2 + 50))
    scrn.blit(restart_text, restart_text_rect)

def restart():
    global bird_y, bird_change_y, score, pipe_x, pipe_height, game_over,score,pipe_speed
    bird_y =s_height // 2 
    bird_change_y= 0
    score= 0
    pipe_x= s_width
    pipe_height= random.randint(200, 400)
    game_over= False
    score=0
    pipe_speed=6

font=pygame.font.SysFont(None,50)

def menu():
    
    text = font.render("FLAPPY BIRD",True,black)
    start_text = text.get_rect(center=(s_width//2,s_height//4))
    scrn.blit(text,start_text)

    text = font.render("1.Press space bar to fly. ",True,black)
    start_text = text.get_rect(center=(s_width//2,s_height//2 + 50))
    scrn.blit(text,start_text)

    text = font.render("2.Avoid hitting the pipes",True,black)
    start_text = text.get_rect(center=(s_width//2,s_height//2 +100))
    scrn.blit(text,start_text)

    text = font.render("3.Fly as long as you can.",True,black)
    start_text = text.get_rect(center=(s_width//2,s_height//2 +150))
    scrn.blit(text,start_text)
                      
# Game loop
running = True
bird_change_y = 0
game_over = False
menu_display = True
score=0

while running:
    scrn.fill(blue)

    if menu_display:
        menu()
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                runnung = False
            elif event.type==pygame.MOUSEBUTTONDOWN:
                menu_display = False
                break
        continue

    for event in pygame.event.get():
       
        if event.type==pygame.QUIT:
            running=False
        if game_over and event.type==pygame.MOUSEBUTTONDOWN:
            restart()
        if not game_over and event.type==pygame.KEYDOWN:
            if event.key==pygame.K_SPACE:
                bird_change_y=bird_jump

    # To update the bird position
    if not game_over:
        bird_y += bird_change_y
        bird_change_y += gravity

    # drawing the bird
    draw_bird(bird_x, bird_y)

    # To update the pipe position
    if not game_over:
        pipe_x -= pipe_speed
        draw_pipe(pipe_x, pipe_height)
        pipe_x -= pipe_speed
       

    # Checking the collision
    if (bird_y > s_height or bird_y < 0 or check_collision()) and not game_over:
        game_over = True
    # To increment the score and pipe speed
    if pipe_x < -pipe_width:
        pipe_x = s_width
        reset_pipe()
        if not game_over:
            scores.append(score)
            score += 1
            pipe_speed += 0.2

    display_score(score)

    # To show the final score
    if game_over:
        final_score(score)

    # To update display
    pygame.display.update()

    # To control the fps
    clock.tick(60)

# Function to quit pygame
pygame.quit()