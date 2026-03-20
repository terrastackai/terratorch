# Copyright contributors to the Terratorch project

from collections.abc import Iterable, Iterator
from random import choice

from torch.utils.data import Sampler
from torchgeo.datasets import BoundingBox
from torchgeo.samplers import GeoSampler


class MultiSampler(Sampler[BoundingBox]):
    def __init__(self, samplers: Iterable[GeoSampler]) -> None:
        self.length = sum(map(len, samplers))
        self.samplers = samplers

    def __len__(self) -> int:
        return self.length


class RandomMultiSampler(MultiSampler):
    def __iter__(self) -> Iterator[BoundingBox]:
        sampler_set = set(list(iter(sampler) for sampler in self.samplers))
        sampler_list = list(sampler_set)
        while len(sampler_set) > 0:
            sampler = choice(sampler_list)
            try:
                yield next(sampler)
            except StopIteration:
                sampler_set.remove(sampler)
                sampler_list = list(sampler_set)


class SequentialMultiSampler(MultiSampler):
    def __iter__(self) -> Iterator[BoundingBox]:
        sampler_iterables = list(iter(sampler) for sampler in self.samplers)
        for sampler in sampler_iterables:
            for sample in sampler:
                yield sample
