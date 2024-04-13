import pygame
from run import GameController
from constants import *
import pygame
import neat
import os
import time
import pickle
import keyboard


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

        while run:
            dt = self.clock.tick(30) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

            # Check if Pac-Man's position matches any of the node positions
            pacman_pos = (self.pacman.position.x, self.pacman.position.y)
            for node_pos in self.turning_points:
                if pacman_pos == node_pos:
                    # Feed Pac-Man's position and other relevant data to the neural network
                    output1 = net1.activate((self.pacman.position.x, self.pacman.position.y,
                                             self.ghosts.blinky.position.x, self.ghosts.blinky.position.y,
                                             self.ghosts.clyde.position.x, self.ghosts.clyde.position.y,
                                             self.ghosts.inky.position.x, self.ghosts.inky.position.y,
                                             self.ghosts.pinky.position.x, self.ghosts.pinky.position.y,
                                             self.game.lives))

                    # Determine the decision based on the neural network output
                    decision = output1.index(max(output1))
                    if decision == 1:
                        keyboard.press('w')
                    elif decision == 2:
                        keyboard.press('s')
                    elif decision == 3:
                        keyboard.press('a')
                    elif decision == 4:
                        keyboard.press('d')
            # print(output1,output2)
            # decision = output1.index(max(output1))
            # print(decision)
            game_info = self.game.loop()
            self.game.render()
            pygame.display.update()

            if game_info.lives < 5:
                self.calculate_fitness(genome1, genome2, game_info)
                break

    def calculate_fitness(self, genome1, genome2, game_info):
        genome1.fitness += self.game.score
        genome2.fitness += self.game.score
        pass

    def move_ai(self, net):
        pass

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

    pygame.display.set_caption("Pong")
    pac_game = GameController()
    pac_game.test_ai(winner_net)


def run_neat(config):
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-85')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 50)
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
