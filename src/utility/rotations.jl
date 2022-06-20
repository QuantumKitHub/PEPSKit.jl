const NORTH = 1;
const EAST = 2;
const SOUTH = 3;
const WEST = 4;

const NORTHWEST = 1;
const NORTHEAST = 2;
const SOUTHEAST = 3;
const SOUTHWEST = 4;

rotate_north(t,dir) = mod1(dir,4) == NORTH ? t : rotate_north(rotl90(t),dir-1)

#Base.rotl90(t::Tuple) = rotl90.(t);
