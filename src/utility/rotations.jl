#direction i -> the direction that the peps leg i points to

const Dir = Int64; #initially I had Dir = Uint8; but space(TensorMap,Uint8) errors

const North =   4 :: Dir;
const East  =   3 :: Dir;
const South =   2 :: Dir;
const West  =   1 :: Dir;

const NorthWest = North;
const NorthEast = East;
const SouthEast = South;
const SouthWest = West;

const Dirs = [West,South,East,North];

Base.rotl90(dir::Dir) = Dir(mod1(dir+1,4))
Base.rotr90(dir::Dir) = Dir(mod1(dir-1,4))

left(dir::Dir) = rotl90(dir);
right(dir::Dir) = rotr90(dir);

#allow rotations of single peps tensors
Base.rotl90(t::PEPSType) = permute(t,(4,1,2,3),(5,))
Base.rotr90(t::PEPSType) = permute(t,(2,3,4,1),(5,))

#allow rotations of effective h - n matrices
Base.rotl90(t::EffM) = permute(t,(4,1,2,3,5),(9,6,7,8,10))
Base.rotr90(t::EffM) = permute(t,(2,3,4,1,5),(7,8,9,6,10))

#allow rotations of (numrows,numcols)
Base.rotl90(bbox::Tuple{Int,Int}) = reverse(bbox);
Base.rotr90(bbox::Tuple{Int,Int}) = reverse(bbox);

function rotate_north(data,dir::Dir)
    if dir == North
        return data
    end

    return rotate_north(rotl90(data),rotl90(dir))
end

inv_rotate_north(dat,dir) = rotate_north(dat,Dir(mod1(-dir,4)))

rotate_north(coords::Tuple{Int,Int},bbox::Tuple{Int,Int},dir) = (dir == North) ? coords : rotate_north((coords[2],bbox[1]-coords[1]+1),reverse(bbox),right(dir))
inv_rotate_north(coords::Tuple{Int,Int},bbox::Tuple{Int,Int},dir) = rotate_north(coords,rotate_north(bbox,dir),Dir(mod1(-dir,4)))
