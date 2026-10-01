from typing import Any

import blokudoku_lib as bl
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

IN_CONV_CHANNELS = 1
OUT_CONV_CHANNELS_1 = 32
OUT_CONV_CHANNELS_2 = 64
OUT_CONV_CHANNELS_3 = 64
OUT_CONV_CHANNELS_4 = 16
KERNEL_SIZE = (3, 3)
PADDING = 1

OUT_EMBEDDING_DIM = 16

IN_LINEAR_DIM = (OUT_CONV_CHANNELS_4 * bl.BOARD_FLAT) + (OUT_EMBEDDING_DIM * bl.BLOCKS_TO_PICK) + 1
HID_LINEAR_DIM = 256
OUT_LINEAR_DIM = bl.BLOCKS_TO_PICK * bl.BOARD_FLAT

class BLOCK_Q_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(IN_CONV_CHANNELS, OUT_CONV_CHANNELS_1, KERNEL_SIZE, padding=PADDING),
            nn.ReLU(),
            nn.Conv2d(OUT_CONV_CHANNELS_1, OUT_CONV_CHANNELS_2, KERNEL_SIZE, padding=PADDING),
            nn.ReLU(),
            nn.Conv2d(OUT_CONV_CHANNELS_2, OUT_CONV_CHANNELS_3, KERNEL_SIZE, padding=PADDING),
            nn.ReLU(),
            nn.Conv2d(OUT_CONV_CHANNELS_3, OUT_CONV_CHANNELS_4, KERNEL_SIZE, padding=PADDING),
            nn.ReLU(),
            nn.Flatten()
        )

        self.embed_net = nn.Sequential(
            nn.Embedding(bl.BLOCKS_SIZE + 1, OUT_EMBEDDING_DIM),
            nn.Flatten()
        )

        self.afterwards_net = nn.Sequential(
            nn.Linear(IN_LINEAR_DIM, HID_LINEAR_DIM),
            nn.ReLU(),
            nn.Linear(HID_LINEAR_DIM, HID_LINEAR_DIM),
            nn.ReLU(),
            nn.Linear(HID_LINEAR_DIM, OUT_LINEAR_DIM),
        )

    def forward(self, board: torch.Tensor, blocks: torch.Tensor, streak: torch.Tensor):
        board_out = self.conv_net(board)
        blocks_out = self.embed_net(blocks)

        final_input = torch.cat((board_out, blocks_out, streak), dim=1)
        return self.afterwards_net(final_input)

def train():
    pass

def main(args):
    if len(args) != 2:
       print(f"Main needs one argument, but received: {len(args) - 1} arguments!")
       return 0

    if args[1] == "train":
        train()
    elif args[1] == "test":
        raise RuntimeError("Not implemented yet!")
    elif args[1] == "play_as_player":
        raise RuntimeError("Not implemented yet!")
    elif args[1] == "watch_ai_play":
        raise RuntimeError("Not implemented yet!")
    elif args[1] == "usage":
        print(
    """Commands available for blokudoku_AI:
    train: trains the AI model
    test: tests the AI model
    play_as_player: play the game as a player, to compare score with the AI
    watch_ai_play: AI plays slowly, so you can watch it play
    """)
    else:
        print(f"Unrecgonised argument: {args[1]}! Use command 'usage' for help! ")
        return 0


if __name__ == "__main__":
    main(sys.argv)
