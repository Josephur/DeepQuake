"""A minimal Quake map, generated at run time.

For tests that need a map which is not id's (#28): no map file may be
committed (it would be game data), so this module writes the bytes of a
small, valid BSP 29 file — three rooms joined by doorways, an
``info_player_start``, nothing else. The server only ever reads the
entity, plane, node, clipnode, leaf and model lumps; texture, lighting,
visibility and surface lumps are empty, which the loader accepts.

The collision trees are built for points (hull 0) and for the two player
hulls, whose brushes are the rooms expanded by the hull size — qbsp does
this when it compiles a real map; the loader does not.
"""

from __future__ import annotations

import struct

BSPVERSION = 29
CONTENTS_EMPTY = -1
CONTENTS_SOLID = -2

# Three rooms in a row, joined by doorways that pierce the 48-unit walls.
ROOMS = [
    ((-536.0, -128.0, 0.0), (-24.0, 128.0, 192.0)),
    ((24.0, -128.0, 0.0), (520.0, 128.0, 192.0)),
    ((568.0, -128.0, 0.0), (1064.0, 128.0, 192.0)),
]
DOORS = [
    ((-64.0, -32.0, 0.0), (64.0, 32.0, 72.0)),
    ((480.0, -32.0, 0.0), (608.0, 32.0, 72.0)),
]
SPAWN = (-280.0, 0.0, 24.0)  # the info_player_start, on room A's floor
WORLD = 16384.0  # everything outside the rooms is solid out to this bound

# (mins, maxs) of the player's two collision hulls, from Mod_LoadClipnodes.
HULLS = (
    ((-16.0, -16.0, -24.0), (16.0, 16.0, 32.0)),
    ((-32.0, -32.0, -24.0), (32.0, 32.0, 64.0)),
)

_Box = tuple[tuple[float, float, float], tuple[float, float, float]]
_REGION: _Box = ((-WORLD, -WORLD, -WORLD), (WORLD, WORLD, WORLD))


def _overlap(box: _Box, region: _Box) -> bool:
    return all(box[1][a] > region[0][a] and box[0][a] < region[1][a] for a in range(3))


def _contains(box: _Box, region: _Box) -> bool:
    return all(box[0][a] <= region[0][a] and box[1][a] >= region[1][a] for a in range(3))


def _split(region: _Box, axis: int, value: float) -> tuple[_Box, _Box]:
    """The region below and above a cut at ``value`` along ``axis``."""
    back_lo, back_hi = [list(c) for c in region]
    front_lo, front_hi = [list(c) for c in region]
    back_hi[axis] = front_lo[axis] = value
    return (tuple(back_lo), tuple(back_hi)), (tuple(front_lo), tuple(front_hi))  # type: ignore


def _tree(region: _Box, boxes: list[_Box]):
    """A kd-tree over the union of ``boxes``: leaves are 'solid' or 'empty'.

    Splits only at box faces strictly inside the region, so every recursion
    removes one candidate face and the tree terminates with each leaf fully
    inside or fully outside every box.
    """
    boxes = [box for box in boxes if _overlap(box, region)]
    if not boxes:
        return ("solid",)
    if any(_contains(box, region) for box in boxes):
        return ("empty",)
    cuts = [
        (axis, value)
        for box in boxes
        for axis in range(3)
        for value in (box[0][axis], box[1][axis])
        if region[0][axis] < value < region[1][axis]
    ]
    axis, value = cuts[len(cuts) // 2]  # the median keeps the tree shallow
    back, front = _split(region, axis, value)
    return ("node", axis, value, _tree(back, boxes), _tree(front, boxes))


def _expand(box: _Box, hull: _Box) -> _Box:
    """The clip-hull brush: a point in it collides with the room as a body."""
    mins = tuple(box[0][a] - hull[1][a] for a in range(3))
    maxs = tuple(box[1][a] - hull[0][a] for a in range(3))
    return (mins, maxs)  # type: ignore[return-value]


def _leaf_regions(tree, region: _Box, label: str) -> list[_Box]:
    """The regions of the kd partition whose leaf is ``label``, as boxes.

    The 'solid' leaves of the playable-space tree are the walls (everything
    outside the rooms and doorways) cut into boxes — the brushes a real map's
    clip hulls are built from.
    """
    if tree[0] != "node":
        return [region] if tree[0] == label else []
    _, axis, value, back, front = tree
    back_region, front_region = _split(region, axis, value)
    return _leaf_regions(back, back_region, label) + _leaf_regions(front, front_region, label)


class _Bsp:
    """Collects planes and leaves, and emits one tree as nodes, others as clipnodes.

    Tree children are encoded exactly as the loaders decode them: node
    children name a node index or ``65535 - leaf``; clipnode children name a
    clipnode index or the contents as an unsigned 16-bit value.
    """

    def __init__(self) -> None:
        self.planes: list[tuple[int, float]] = []  # (axis, distance)
        self.leaves = [CONTENTS_SOLID]  # leaf 0: the generic solid leaf
        self.nodes: list[dict] = []
        self.clipnodes: list[dict] = []
        self.head_node = 0
        self.head_clipnodes = [0, 0]

    def plane(self, axis: int, value: float) -> int:
        entry = (axis, value)
        if entry not in self.planes:
            self.planes.append(entry)
        return self.planes.index(entry)

    def emit_nodes(self, tree, region: _Box) -> int:
        """Add a hull-0 subtree in pre-order; returns its root's node index.

        The engine walks these trees starting from the root and requires
        every child to lie at a higher index than its parent (and hull 0's
        root to be node 0: SV_PointContents starts there unconditionally),
        so the parent is numbered before its children.
        """
        if tree[0] != "node":
            leaf = (
                0
                if tree[0] == "solid"
                else self.leaves.append(CONTENTS_EMPTY) or len(self.leaves) - 1
            )
            return 65535 - leaf
        _, axis, value, back, front = tree
        index = len(self.nodes)
        node = {"plane": self.plane(axis, value), "children": [0, 0], "region": region}
        self.nodes.append(node)
        back_region, front_region = _split(region, axis, value)
        node["children"] = [
            self.emit_nodes(front, front_region),  # children[0] is the front
            self.emit_nodes(back, back_region),
        ]  # side, as the engine decodes it: the normal is +axis, so front is up
        return index

    def emit_clipnodes(self, tree, region: _Box) -> int:
        """Add a clip-hull subtree in pre-order; returns a clipnode index.

        The clip-hull boxes are the wall brushes *expanded by the hull size*,
        so they are where the player's origin may NOT go: inside a box is
        solid, and the free space is everything outside them — the leaf
        labels come out opposite to the hull-0 tree, which marks the rooms.
        """
        if tree[0] != "node":
            return CONTENTS_SOLID if tree[0] == "empty" else CONTENTS_EMPTY
        _, axis, value, back, front = tree
        index = len(self.clipnodes)
        node = {"plane": self.plane(axis, value), "children": [0, 0]}
        self.clipnodes.append(node)
        back_region, front_region = _split(region, axis, value)
        node["children"] = [
            self.emit_clipnodes(front, front_region),
            self.emit_clipnodes(back, back_region),
        ]
        return index

    def bytes(self) -> bytes:
        """The whole file: a 124-byte header, then lumps in index order."""
        mins = tuple(min(r[0][a] for r in ROOMS + DOORS) for a in range(3))
        maxs = tuple(max(r[1][a] for r in ROOMS + DOORS) for a in range(3))
        planes = b"".join(
            struct.pack(
                "<ffffi",
                *(float(axis == a) for a in range(3)),
                value,
                axis,  # PLANE_X/Y/Z: the axial plane types are 0, 1 and 2
            )
            for axis, value in self.planes
        )

        def short(value: float) -> int:
            return int(max(-32768.0, min(32767.0, round(value))))

        nodes = b"".join(
            struct.pack(
                "<iHHhhhhhhHH",
                node["plane"],
                *node["children"],  # already 65535 - leaf, or a node index
                *(short(v) for v in node["region"][0]),
                *(short(v) for v in node["region"][1]),
                0,
                0,  # no faces
            )
            for node in self.nodes
        )
        clipnodes = b"".join(
            struct.pack(
                "<iHH",
                node["plane"],
                *(child + 65536 if child < 0 else child for child in node["children"]),
            )
            for node in self.clipnodes
        )
        leaves = b"".join(
            struct.pack(
                "<iihhhhhhHH4B",
                contents,
                -1,  # visofs: no visibility info
                short(-WORLD),
                short(-WORLD),
                short(-WORLD),
                short(WORLD),
                short(WORLD),
                short(WORLD),
                0,
                0,  # no mark surfaces
                0,
                0,
                0,
                0,  # ambient sound levels
            )  # fmt: skip
            for contents in self.leaves
        )
        entities = "".join(
            [
                '{\n"classname" "worldspawn"\n',
                '"message" "generator test map"\n',
                "}\n",
                '{\n"classname" "info_player_start"\n',
                f'"origin" "{SPAWN[0]:.0f} {SPAWN[1]:.0f} {SPAWN[2]:.0f}"\n',
                '"angle" "90"\n',
                "}\n",
                *[f'{{\n"classname" "light"\n"origin" "{x} 0 160"\n}}\n' for x in (-280, 272, 816)],
            ]
        ).encode("ascii")
        models = struct.pack(
            "<3f3f3f4iiii",
            *mins,
            *maxs,
            0.0,
            0.0,
            0.0,
            self.head_node,
            self.head_clipnodes[0],
            self.head_clipnodes[1],
            self.head_clipnodes[1],  # hull 3 is unused; point it at hull 2
            len(self.leaves) - 1,  # visible leaves: all but the solid one
            0,
            0,  # no faces
        )

        lumps = [entities, planes, b"", b"", b"", nodes, b"", b"", b"", clipnodes, leaves,
                 b"", b"", b"", models]  # fmt: skip
        header = bytearray(struct.pack("<i", BSPVERSION) + b"\0" * (15 * 8))
        body = bytearray()
        for index, data in enumerate(lumps):
            pad = -len(body) % 4  # every lump stays 4-byte aligned
            struct.pack_into("<ii", header, 4 + 8 * index, len(header) + len(body) + pad, len(data))
            body += b"\0" * pad + data
        return bytes(header + body)


def generate() -> bytes:
    """The map file: three rooms, two doorways, one spawn point."""
    bsp = _Bsp()
    playable = ROOMS + DOORS
    hull0 = _tree(_REGION, playable)
    bsp.head_node = bsp.emit_nodes(hull0, _REGION)
    walls = _leaf_regions(hull0, _REGION, "solid")
    bsp.head_clipnodes = [
        bsp.emit_clipnodes(_tree(_REGION, [_expand(box, hull) for box in walls]), _REGION)
        for hull in HULLS
    ]
    return bsp.bytes()
