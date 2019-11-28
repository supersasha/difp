import numpy as np
import argparse

from profile import FilmProfile
from brewer import brewer

def arguments():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('film_datasheet')
    parser_test.add_argument('paper_datasheet')
    
    return parser.parse_args()

if __name__ == '__main__':
    opts = arguments()
    film = FilmProfile(opts.film_datasheet, mode31=True)
    paper = FilmProfile(opts.paper_datasheet, mode31=True)
    paper_gammas = np.array([4.0, 4.0, 4.0])
    sense = film.sense()
    dyes = paper.dye()
    print(np.argmax(dyes, axis=1))
    print(np.max(dyes, axis=1))
    brwr = brewer(sense, dyes)
    neg_gammas = brwr.Gammas.transpose().dot(np.diag(1/paper_gammas))
    print('Brewer gammas:', brwr.Gammas)
    print('Negative gammas:', neg_gammas)

