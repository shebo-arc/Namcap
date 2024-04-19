import pygame
from run import GameController
from constants import *
import pygame
import neat
import os
import time
import pickle
import keyboard
import math
import datetime


class pacman_game:
    def __init__(self):
        self.game = GameController()
        self.game.startGame()
        self.pacman = self.game.pacman
        self.ghosts = self.game.ghosts  # list
        self.score = self.game.score
        self.clock = pygame.time.Clock()
        self.turning_points = self.game.nodes.nodesLUT.keys()
        self.intersection_points = [[], []]
        for node in self.turning_points:
            self.intersection_points[0].append(node[0])
            self.intersection_points[1].append(node[1])

            # (node.position.x, node.position.y)
        print(self.intersection_points[0])
        print(self.intersection_points[1])

        print("pacman at")
        print(self.pacman.position)

    def test_ai(self):
        run = True
        # clock = pygame.time.Clock()
        self.game.startGame()
        while run:
            # clock.tick(600)

            self.game.loop()
            self.game.render()
            pygame.display.update()
            game_info = self.game.loop()
            # print(game_info.pacman_pos) #prints pacmans coordinates on the board
            # print(game_info.ghosts_pos) #prints all ghosts coordinates on the board

    def train_ai(self, genome1, genome2, config):
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

        run = True
        self.game.startGame()
        # self.pause.setPause(playerPaused=True)
        self.game.pause.setPause(playerPaused=False)
        time_stamp = None
        last_position = None  # Store the last position of the agent
        last_position_time = None

        while run:
            dt = self.clock.tick(300) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
            self.pacman.update(dt)
            game_info = self.game.loop()
            self.game.render()
            pygame.display.update()

            # Check if Pac-Man's position matches any of the node positions
            pacman_pos = (game_info.pacman_pos_x.x, game_info.pacman_pos_x.y)
            # print(pacman_pos)
            current_position = (game_info.pacman_pos_x.x, game_info.pacman_pos_x.y)

            if current_position != last_position:
                last_position = current_position
                current_time = datetime.datetime.now()
                time_string = current_time.strftime("%H%M%S")
                last_position_time = time_string
                # print(last_position)
                # last_position_time = time.time()
            elif current_position == last_position:
                current_time = datetime.datetime.now()
                time_string = current_time.strftime("%H%M%S")
                if int(time_string) - int(last_position_time) > 1.5:
                    self.calculate_fitness(genome1, genome2, game_info)
                    break

            for node_pos in self.turning_points:
                if pacman_pos == node_pos:
                    print("intersection")
                    # self.move_ai(net1,game_info)

                    # print("intersection")
                    current_time = datetime.datetime.now()
                    time_string = current_time.strftime("%H%M%S")  # Format as string
                    print("Current time:", time_string)
                    output1 = net1.activate((game_info.pacman_pos_x.x, game_info.pacman_pos_x.y,
                                             game_info.ghosts_pos[0].x - game_info.pacman_pos_x.x,
                                             game_info.ghosts_pos[0].y - game_info.pacman_pos_x.y,
                                             game_info.ghosts_pos[1].x - game_info.pacman_pos_x.x,
                                             game_info.ghosts_pos[1].y - game_info.pacman_pos_x.y,
                                             game_info.ghosts_pos[2].x - game_info.pacman_pos_x.x,
                                             game_info.ghosts_pos[2].y - game_info.pacman_pos_x.y,
                                             game_info.ghosts_pos[3].x - game_info.pacman_pos_x.x,
                                             game_info.ghosts_pos[3].y - game_info.pacman_pos_x.y,
                                             game_info.score
                                             ))

                    # print(game_info.ghosts_pos[0].x)
                    # print(self.pacman.position.x)
                    # print("output done")

                    # Feed Pac-Man's position and other relevant data to the neural network

                    # Determine the decision based on the neural network output
                    decision = output1.index(max(output1))
                    if decision == 0:
                        # self.game.pacman.direction = UP
                        keyboard.press('w')
                        keyboard.release('a')
                        keyboard.release('s')
                        keyboard.release('d')
                        # new_time = datetime.datetime.now()
                        # new_time_string = new_time.strftime("%H%M%S")
                        # time_stamp = new_time_string
                        print("up")

                    elif decision == 1:
                        keyboard.press('s')
                        keyboard.release('d')
                        keyboard.release('a')
                        keyboard.release('w')
                        print("down")

                    elif decision == 2:
                        keyboard.press('a')
                        keyboard.release('d')
                        keyboard.release('s')
                        keyboard.release('w')
                        print("left")

                    elif decision == 3:
                        keyboard.press('d')
                        keyboard.release('s')
                        keyboard.release('a')
                        keyboard.release('w')

                        print("right")
                else:
                    pass
                self.pacman.update(dt)
            if current_position != last_position:
                last_position = current_position
                # last_position_time = time.time()

            # print(output1,output2)
            # decision = output1.index(max(output1))
            # print(decision)
            game_info = self.game.loop()
            self.game.render()
            pygame.display.update()

            # print(time.time())
            # Format as string

            if game_info.lives < 5:
                self.calculate_fitness(genome1, genome2, game_info)
                break

    def calculate_fitness(self, genome1, genome2, game_info):
        genome1.fitness += game_info.score + game_info.time_elapsed / 1000
        genome2.fitness += 0
        print(genome1.fitness)

    def move_ai(self, net, game_info):
        output1 = net.activate((game_info.pacman_pos_x.x, game_info.pacman_pos_x.y,
                                game_info.ghosts_pos[0].x - game_info.pacman_pos_x.x,
                                game_info.ghosts_pos[0].y - game_info.pacman_pos_x.y,
                                game_info.ghosts_pos[1].x - game_info.pacman_pos_x.x,
                                game_info.ghosts_pos[1].y - game_info.pacman_pos_x.y,
                                game_info.ghosts_pos[2].x - game_info.pacman_pos_x.x,
                                game_info.ghosts_pos[2].y - game_info.pacman_pos_x.y,
                                game_info.ghosts_pos[3].x - game_info.pacman_pos_x.x,
                                game_info.ghosts_pos[3].y - game_info.pacman_pos_x.y,
                                game_info.score
                                ))
        decision = output1.index(max(output1))
        if decision == 0:
            # self.game.pacman.direction = UP
            keyboard.press('w')
            keyboard.release('a')
            keyboard.release('s')
            keyboard.release('d')
            print("up")

        elif decision == 1:
            keyboard.press('s')
            keyboard.release('d')
            keyboard.release('a')
            keyboard.release('w')
            print("down")

        elif decision == 2:
            keyboard.press('a')
            keyboard.release('d')
            keyboard.release('s')
            keyboard.release('w')
            print("left")

        elif decision == 3:
            keyboard.press('d')
            keyboard.release('s')
            keyboard.release('a')
            keyboard.release('w')

            print("right")
        pass


# players = [(self.genome1, net, self.left_paddle, True)]
# for (genome, net, paddle, left) in players:
# output = net.activate(
# (paddle.y, abs(paddle.x - self.ball.x), self.ball.y))
# decision = output.index(max(output))

# def calculate_fitness(self, game_info, duration):
# self.genome.fitness += game_info.score + duration


def eval_genomes(genomes, config):
    """
    Run each genome against eachother one time to determine the fitness.
    """
    for i, (genome_id1, genome1) in enumerate(genomes):
        print(round(i / len(genomes) * 100), end=" ")
        genome1.fitness = 0
        for genome_id2, genome2 in genomes[min(i + 1, len(genomes) - 1):]:
            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness
            game = pacman_game()
            game.train_ai(genome1, genome2, config)

            pass


def test_best_network(config):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    #pygame.display.set_caption("Pong")
    pac_game = GameController()
    pac_game.test_ai(winner_net)


def run_neat(config):
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-1')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 50)
    print(stats)
    # with open("best.pickle", "wb") as f:
    # pickle.dump(winner, f)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    run_neat(config)
    # test_best_network(config)
