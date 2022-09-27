//! 2D non-balanced B-tree of bounded depth, for storing integer rectangles, and doing
//! efficient searches through them.
#![deny(warnings)]
#![deny(missing_docs)]
#![feature(int_log)]
#![feature(stdsimd)]
#![feature(bench_black_box)]
#![feature(test)]

#[macro_use]
extern crate savefile_derive;
use indexmap::IndexMap;
use std::ops::{Sub, Add};
use smallvec::smallvec;

use std::hash::Hash;
use std::arch::x86_64::{__m128i, _mm_set1_epi8, _mm_cvtsi128_si64x, _mm_and_si128, _mm_set1_epi64x};

#[cfg(all(target_feature = "avx512vl", target_feature="avx512bw"))]
use std::arch::x86_64::{_mm_broadcastb_epi8, _mm_maskz_broadcastb_epi8};
use smallvec::SmallVec;
use savefile::{WithSchema, Serialize, Deserialize, Introspect};

/// A coordinate point in the 2D world
#[derive(Debug,Clone,Copy,PartialEq,Eq,Hash,Savefile)]
pub struct Pos {
    /// The x-coordinate
    pub x: i32,
    /// The y-coordinate
    pub y: i32,
}

/// Bounding box for an item. The bounding box includes
/// both the top_left corner and the bottom right corner.
/// An object within a bounding box is contained entirely within
/// the box.
#[derive(Debug,Clone,Copy,PartialEq,Eq,Hash,Savefile)]
pub struct BB {
    /// The top left corner of the bounding box. The corner itself is considered part
    /// of the box.
    pub top_left: Pos,
    /// The bottom right corner of the bounding box. The corner itself is considered part
    /// of the box. I.e, this is an inclusive bound.
    pub bottom_right: Pos,
}

impl BB {

    /// Return a new BB, which is 'amount' larger in every direction
    pub fn grow(self, amount: i32) -> BB {
        BB {
            top_left: self.top_left - Pos{x: amount, y:amount},
            bottom_right: self.bottom_right + Pos{x: amount, y:amount},
        }
    }

    /// Return the distance between the two closest points on the two bounding boxes, according to
    /// a uniform norm/infinity norm. I.e, the number of steps which must be walked if
    /// one can walk up, down, left, right or diagonally.
    ///
    /// The points (1,1) and (3,3) have a distance of 2. As have (1,1) and (1,3).
    pub fn distance_to(self, other:BB) -> i32 {
        let x_dist;
        if self.bottom_right.x < other.top_left.x {
            x_dist = other.top_left.x - self.bottom_right.x;
        } else if other.bottom_right.x < self.top_left.x {
            x_dist = self.top_left.x - other.bottom_right.x;
        }  else {
            x_dist = 0;
        }

        let y_dist;
        if self.bottom_right.y < other.top_left.y {
            y_dist = other.top_left.y - self.bottom_right.y;
        } else if other.bottom_right.y < self.top_left.y {
            y_dist = self.top_left.y - other.bottom_right.y;
        }  else {
            y_dist = 0;
        }

        x_dist.max(y_dist)
    }
    /// Create a new bounding box from the given coordinates.
    /// x1,y1 must be the top left corner, and x2,y2 must be the
    /// lower right.
    pub const fn new(x1:i32, y1:i32, x2:i32, y2:i32) -> BB {
        BB {
            top_left: Pos {
                x: x1,
                y: y1
            },
            bottom_right: Pos {
                x: x2,
                y: y2,
            }
        }
    }

    /// Return true if the two boxes overlap. They must actually
    /// overlap, it is not enough that they are just adjacent.
    #[inline(always)]
    pub fn overlaps(self, other: BB) -> bool {
        if  self.top_left.x > other.bottom_right.x ||
            self.top_left.y > other.bottom_right.y ||
            self.bottom_right.x < other.top_left.x ||
            self.bottom_right.y < other.top_left.y {
            false
        } else {
            true
        }
    }

    /// Returns true if the self BB is entirely contained within the 'other'.
    pub fn is_contained_in(self, other: BB) -> bool {
        self.top_left.x >= other.top_left.x &&
        self.top_left.y >= other.top_left.y &&
        self.bottom_right.x <= other.bottom_right.x &&
        self.bottom_right.y <= other.bottom_right.y
    }
}

impl Sub for Pos {
    type Output = Pos;

    fn sub(self, rhs: Self) -> Self::Output {
        Pos {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl Add for Pos {
    type Output = Pos;

    fn add(self, rhs: Self) -> Self::Output {
        Pos {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Sub<Pos> for BB {
    type Output = BB;

    fn sub(self, rhs: Pos) -> Self::Output {
        BB {
            top_left: self.top_left - rhs,
            bottom_right: self.bottom_right - rhs,
        }
    }
}

/// Trait which must be implemented by objects which are to be stored
/// in the QuadBTree map.
pub trait TreeNodeItem {
    /// The type of a unique key identifying a specific item.
    /// If you don't need this, you can use the BB as the key,
    /// or even Self.
    type Key : PartialEq+Eq+Hash+WithSchema+Serialize+Deserialize+Introspect;

    /// Return bounding box for the given item
    fn get_bb(&self) -> BB;

    /// Return a key identifying a particular item.
    fn get_key(&self) -> Self::Key;

    /// Must update the bounding box of the object to the given bounding box
    fn move_item(&mut self, new_bb: BB);

}

fn get_set_bits_below(bitmap:u64, bit_index: u64) -> u32 {
    let below_mask = (1<<bit_index) - 1;
    (bitmap & below_mask).count_ones()
}

#[derive(Savefile,Clone,Debug)]
struct TreeNode<T:WithSchema + Serialize + Deserialize+Introspect> {
    bb: BB,
    sub_cell_size: i32,
    sub_cell_shift: u32,
    /// Terminology:
    /// This bitmap contains 64 bits. Each bit corresponds to a 'place'.
    /// The places are the 64 cells in the 8x8 grid of this node. I.e,
    /// the place number has geometric interpretation.
    /// The numbering is in row-major order.
    node_bitmap: u64,
    /// Actual payload items
    node_payload: SmallVec<[T;4]>,
    /// Terminology:
    /// Each TreeNode can at most have 64 children. Typically, not all are populated.
    /// Each element in 'node_children' is a 'slot'. Each slot is identified by its 'slot_index'.
    node_children: SmallVec<[u32;4]>,
    /// Terminology:
    /// The tree nodes are identified by an 'index'.
    /// Root node always has index 0.
    /// If TreeNode is part of free-list, this is the next free
    next_free: Option<u32>,
    /// The root has parent id 0. It doesn't really have a parent of its own, but it has 0 here.
    parent: u32,
    /// The place of this node in the 8x8 grid of its parent. Row major order.
    node_place: u32,
}

/// 2D non-balanced B-tree of bounded depth, for storing integer rectangles, and doing
/// efficient searches through them.
/// Items inserted should have unique id:s. Only 2D i32 coordinates are supported.
/// Best performance will be had for items which are *evenly distributed*, and
/// do not each cover large areas. The function which returns overlapping items
/// will have bad performance if the number of overlaps is too large (say more than
/// a few on average per item).
#[derive(Savefile,Clone,Debug)]
pub struct QuadBTree<T:TreeNodeItem + WithSchema + Serialize + Deserialize+Introspect> {
    /// Each position in this list is a node_index.
    tree: Vec<TreeNode<T>>, //The first position is always the root
    first_free: Option<u32>, //If there are free nodes, this is the first free node index.
    items: IndexMap<T::Key, u32>, //map item key to tree node
}

impl<T:TreeNodeItem + WithSchema + Serialize + Deserialize+Introspect + Clone> QuadBTree<T> {
    /// Returns true if an item with the given key was moved to the
    /// new bounding box location. False if item was not found.
    pub fn move_item(&mut self, item: &T) -> bool {
        let key = item.get_key();
        let bb = item.get_bb();
        if let Some(node_index) = self.items.get(&key).copied() {
            let node : &mut TreeNode<T> = &mut self.tree[node_index as usize];
            if bb.is_contained_in(node.bb) {

                let off : BB = bb - node.bb.top_left;
                let x1 = (off.top_left.x>>node.sub_cell_shift).clamp(0,7);
                let x2 = (off.bottom_right.x>>node.sub_cell_shift).clamp(0,7);
                let y1 = (off.top_left.y>>node.sub_cell_shift).clamp(0,7);
                let y2 = (off.bottom_right.y>>node.sub_cell_shift).clamp(0,7);
                if !(node.sub_cell_size >= 8 && x1==x2 && y1==y2) {
                    //No big expensive op needed.
                    for item in node.node_payload.iter_mut() {
                        if item.get_key() == key {
                            item.move_item(bb);
                            return true;
                        }
                    }
                    return true;
                }
            }
            // This could be optimized slightly.
            let item = self.remove(key);
            if let Some(mut item) = item {
                item.move_item(bb);
                self.insert(item);
                true
            } else {
                false
            }
        } else {
            self.insert((*item).clone());
            false
        }
    }

}

impl<T:TreeNodeItem + WithSchema + Serialize + Deserialize+Introspect> QuadBTree<T> {

    /// Iterate over all elements in tree
    pub fn iter(&self) -> impl Iterator<Item=&T> {
        self.tree.iter().flat_map(|x|x.node_payload.iter())
    }

    /// Then number of items in the tree
    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Create a new QuadBTree of the given size.
    /// Only coordinates in the range 0..size are allowed
    /// for objects which are to be put into the tree.
    /// This means that if size is 256, the coordinate (256,0) is _not_valid, but
    /// (255,0) is.
    ///
    /// The smaller the size, the faster performance will be. However, by design, situations
    /// where many items would be expected to occupy the same 8x8 area should be avoided.
    pub fn new(size: i32) -> QuadBTree<T> {
        let shift = (size as u32).ilog2();
        if 1<<shift != size {
            panic!("QuadBTree size must be a power of 2, not {}", size);
        }

        let root = TreeNode {
            bb: BB {
                top_left: Pos{x: 0, y:0},
                bottom_right: Pos{x:size,y:size},
            },
            sub_cell_size: size/8,
            sub_cell_shift: shift.saturating_sub(3),
            node_bitmap: 0,
            node_payload: smallvec![],
            node_children: smallvec![],
            next_free: None,
            parent: 0,
            node_place: 0
        };
        QuadBTree {
            tree: vec![root],
            first_free: None,
            items: IndexMap::new()
        }
    }

    fn remove_node_if_unused(&mut self, node_index: usize) {
        if node_index == 0 {
            return; //Root node is never removed.
        }
        let node : &mut TreeNode<T> = &mut self.tree[node_index];
        let parent_node_index = node.parent as usize;
        if node.node_payload.is_empty() == false || node.node_children.is_empty()==false {
            return;
        }


        debug_assert_eq!(node.node_bitmap,0);
        node.next_free = self.first_free;
        self.first_free = Some(node_index as u32);
        {
            let node_place = node.node_place;
            let parent_node:&mut TreeNode<T> = &mut self.tree[parent_node_index];
            let slot_index = get_set_bits_below(parent_node.node_bitmap, node_place as u64);
            parent_node.node_children.remove(slot_index as usize);
            parent_node.node_bitmap &= !(1u64<< node_place);
            debug_assert_eq!(parent_node.node_bitmap.count_ones() as usize, parent_node.node_children.len());
            self.remove_node_if_unused(parent_node_index);
        }
    }

    /*
    /// This routine returns a bitmask with all place's encompassed by a rectangle
    /// from (x1,y1) to (x2,y2), inclusive. It uses AVX512-magic.
    #[cfg(all(target_feature = "avx512vl", target_feature="avx512bw"))]
    #[inline(always)]
    fn spread_op(x1:i32,y1:i32,x2:i32, y2:i32) -> u64 {
        let each_row = ((2<<(x2 as u8))-1) & ! ((1<<x1)-1);
        let rows_mask = ((2<<(y2 as u8))-1) & ! ((1<<y1)-1);
        unsafe {
            // This is basically severe magic.
            let xtemp =  _mm_set1_epi8(each_row);
            let x = _mm_maskz_broadcastb_epi8(rows_mask, xtemp);
            let x = _mm_cvtsi128_si64x(x) as u64;
            x
        }
    }
*/

    /// This routine returns a bitmask with all place's encompassed by a rectangle
    /// from (x1,y1) to (x2,y2), inclusive.
    #[cfg(not(all(target_feature = "avx512vl", target_feature="avx512bw")))]
    #[inline(always)]
    fn spread_op(x1:i32,y1:i32,x2:i32, y2:i32) -> u64 {
        let each_row = ((2u8<<(x2 as u8)).wrapping_sub(1u8)) & ! ((1<<x1)-1);
        let rows_mask = ((256u64<<(y2 as u64*8)).wrapping_sub(1)) & ! ((1u64<<(y1*8))-1);
        // This is basically magic.
        unsafe {
            let mm_all_rows = _mm_set1_epi8(each_row as i8);
            let mm_rows_mask: __m128i = _mm_set1_epi64x(rows_mask as i64);
            let result = _mm_and_si128(mm_all_rows, mm_rows_mask);
            let x = _mm_cvtsi128_si64x(result) as u64;
            return x;
        }
    }
    /// Find all pairs of objects in 'self' which overlap an object in 'other'.
    /// The same object may be reported as part of many pairs.
    /// Performance is O(N^2) worst case when all objects overlap, and O(N) base case when
    /// none do, given N objects in each tree.
    pub fn find_all_overlapping_neighbors2<'a,F:FnMut(&'a T, &'a T)>(&'a self, other:&'a QuadBTree<T>, mut f: F) {
        //let presently_relevant = vec![0u64;(self.tree.len()+63)/64];
        //let mut relevance_bits: Vec::<u64>::new();

        let mut candidate_stack_a = Vec::new();
        let mut candidate_stack_b = Vec::new();
        let mut temps_a = [vec![],vec![],vec![],vec![],vec![],vec![],vec![],vec![]];
        let mut temps_b = [vec![],vec![],vec![],vec![],vec![],vec![],vec![],vec![]];

        if self.tree[0].bb!=other.tree[0].bb {
            panic!("Both QuadBTree must have identical bounding boxes");
        }

        self.find_all_overlapping_neighbors_impl2(other, 0, 0, &mut candidate_stack_a,&mut candidate_stack_b, &mut f, &mut temps_a, &mut temps_b);
    }
    fn find_all_overlapping_neighbors_impl2<'a,F:FnMut(&'a T, &'a T)>(&'a self, other:&'a QuadBTree<T>, self_node_index: usize, other_node_index: usize, candidate_stack_ref_a: &mut Vec<&'a T>, candidate_stack_ref_b: &mut Vec<&'a T>, f: &mut F, temps_a: &mut [Vec<&'a T>], temps_b: &mut [Vec<&'a T>]) {

        //let candidate_watermark = candidate_stack.len();
        let self_node = &self.tree[self_node_index];
        let other_node = &other.tree[other_node_index];

        debug_assert_eq!(self_node.bb, other_node.bb);

        let (first,second_a) = temps_a.split_at_mut(1);
        let mut candidate_stack_a:&mut Vec<_>;
        candidate_stack_a = &mut first[0];
        candidate_stack_a.clear();
        candidate_stack_a.extend(candidate_stack_ref_a.iter().copied().filter(|x|x.get_bb().overlaps(self_node.bb)));

        let (first,second_b) = temps_b.split_at_mut(1);
        let mut candidate_stack_b:&mut Vec<_>;
        candidate_stack_b = &mut first[0];
        candidate_stack_b.clear();
        candidate_stack_b.extend(candidate_stack_ref_b.iter().copied().filter(|x|x.get_bb().overlaps(self_node.bb)));



        candidate_stack_a.extend(self_node.node_payload.iter());

        for cand in candidate_stack_a.iter().copied()
        {
            for item in other_node.node_payload.iter() {
                if item.get_bb().overlaps(cand.get_bb()) {
                    f(cand, item);
                }
            }
        }

        for cand in candidate_stack_b.iter().copied()
        {
            for item in self_node.node_payload.iter() {
                if item.get_bb().overlaps(cand.get_bb()) {
                    f(item, cand);
                }
            }
        }
        candidate_stack_b.extend(other_node.node_payload.iter());

        for self_child_index in self_node.node_children.iter().copied() {
            let place = self.tree[self_child_index as usize].node_place; //The place is as valid for 'self' as for 'other', since it is a geometric sub place of the current node bounding box
            if other_node.node_bitmap&(1u64<<place) != 0 {
                let other_child_slot_index = get_set_bits_below(other_node.node_bitmap, place as u64);
                let other_child_node_index = other_node.node_children[other_child_slot_index as usize];
                self.find_all_overlapping_neighbors_impl2(other, self_child_index as usize, other_child_node_index as usize, &mut candidate_stack_a, &mut candidate_stack_b, f, second_a, second_b);
            }
        }
    }
    /// Find all pairs of objects which overlap each other.
    /// Each pair (a,b) will only be reported once, not twice as in (a,b) and (b,a).
    /// The order ( (a,b) or (b,a) ) is arbitrary.
    /// The same object may be reported as part of many pairs.
    /// Performance is O(N^2) worst case when all objects overlap, and O(N) base case when
    /// none do.
    pub fn find_all_overlapping_neighbors<'a,F:FnMut(&'a T, &'a T)>(&'a self, mut f: F) {
        //let presently_relevant = vec![0u64;(self.tree.len()+63)/64];
        //let mut relevance_bits: Vec::<u64>::new();

        let mut candidate_stack = Vec::with_capacity(16);
        let mut temps = [vec![],vec![],vec![],vec![],vec![],vec![],vec![],vec![]];

        self.find_all_overlapping_neighbors_impl(0, &mut candidate_stack, &mut f, &mut temps);
    }
    fn find_all_overlapping_neighbors_impl<'a,F:FnMut(&'a T, &'a T)>(&'a self, node_index: usize, candidate_stack_ref: &mut Vec<&'a T>, f: &mut F, temps: &mut [Vec<&'a T>]) {

        //let candidate_watermark = candidate_stack.len();
        let node = &self.tree[node_index];

        let (first,second) = temps.split_at_mut(1);
        let mut candidate_stack:&mut Vec<_>;
        candidate_stack = &mut first[0];
        candidate_stack.clear();
        candidate_stack.extend(candidate_stack_ref.iter().copied().filter(|x|x.get_bb().overlaps(node.bb)));


        let candidate_watermark = candidate_stack.len();
        candidate_stack.extend(node.node_payload.iter());

        let mut diagskip = 0; //Avoid returning each pair twice as (a,b) and (b,a). Also avoid (a,a) :-)
        for (outer_index, cand) in candidate_stack.iter().copied().enumerate()
        {
            if outer_index >= candidate_watermark {
                diagskip += 1;
            }

            for item in node.node_payload.iter().skip(diagskip) {
                if item.get_bb().overlaps(cand.get_bb()) {
                    f(item, cand);
                }
            }

        }
        for child in node.node_children.iter().copied() {
            self.find_all_overlapping_neighbors_impl(child as usize, &mut candidate_stack, f, second);
        }
    }

    fn query_impl<'a,F:FnMut(&'a T)>(&'a self, node_index: usize, bb: BB, f:&mut F) {
        let node = &self.tree[node_index];
        for payload_item in &node.node_payload {
            if bb.overlaps(payload_item.get_bb()) {
                f(payload_item)
            }
        }

        if node.node_bitmap==0 {
            return;
        }

        #[cfg(debug_assertions)]
        if !bb.overlaps(node.bb) {
            debug_assert!(bb.overlaps(node.bb));
        }

        let off : BB = bb - node.bb.top_left;
        let x1 = (off.top_left.x>>node.sub_cell_shift).clamp(0,7);
        let x2 = (off.bottom_right.x>>node.sub_cell_shift).clamp(0,7);
        let y1 = (off.top_left.y>>node.sub_cell_shift).clamp(0,7);
        let y2 = (off.bottom_right.y>>node.sub_cell_shift).clamp(0,7);

        let bb_place_mask = Self::spread_op(x1, y1, x2, y2);
        let mut children_place_mask:u64 = node.node_bitmap&bb_place_mask;
        //compile_error!("Continue optimizing")
        while children_place_mask !=0 {
            let cur_child_place = children_place_mask.trailing_zeros() as u64;
            let cur_child_place_mask = 1<< cur_child_place;
            children_place_mask &= !cur_child_place_mask;
            let child_slot_index = ((cur_child_place_mask - 1)&node.node_bitmap).count_ones();
            let child_node_index = node.node_children[child_slot_index as usize];


            #[cfg(debug_assertions)]
                {
                    let childnode:&TreeNode<T> = &self.tree[child_node_index as usize];
                    debug_assert_eq!(childnode.parent as usize, node_index);
                    let cur_child_x1 = cur_child_place %8;
                    let cur_child_y1 = cur_child_place /8;
                    {
                        debug_assert_eq!(childnode.bb.top_left.x, node.bb.top_left.x + cur_child_x1 as i32*node.sub_cell_size);
                    }
                    {
                        debug_assert_eq!(childnode.bb.top_left.y, node.bb.top_left.y + cur_child_y1 as i32*node.sub_cell_size);
                    }

                }

            self.query_impl(child_node_index as usize, bb, &mut *f);
        }
    }

    /// Find all objects within the given bounding box. Performance is approximately
    /// O(N) where N is the number of items in the box, unless there are many large
    /// objects which almost overlap. This worst case is O(M) where M is the total number
    /// of items in the map.
    ///
    /// This method returns references to all items overlapping the box.
    #[inline(always)]
    pub fn query(&self, bb:BB) -> Vec<&T> {
        let mut ret = vec![];
        self.query_impl(0, bb, &mut |item|ret.push(item));
        ret
    }

    /// Find all objects within the given bounding box. Performance is approximately
    /// O(N) where N is the number of items in the box, unless there are many large
    /// objects which almost overlap. This worst case is O(M) where M is the total number
    /// of items in the map.
    ///
    /// This method calls the provided closure for each item within the bounding box.
    #[inline(always)]
    pub fn query_fn<F:FnMut(&T)>(&self, bb: BB, mut f:F) {
        self.query_impl(0,bb,&mut f)
    }

    /// Get the object with the given key, or None if not found.
    /// Expected performance is O(1) best case, and O(N) worst case. Expect
    /// performance to deteriorate if there are many items nearby the item searched for.
    pub fn get_by_key(&self, key: T::Key) -> Option<&T> {
        if let Some(item) = self.items.get(&key).copied() {
            for payload_item in self.tree[item as usize].node_payload.iter() {
                if payload_item.get_key() == key {
                    return Some(payload_item);
                }
            }
            debug_assert!(false); //We shouldn't ever get here.
            None
        } else {
            None
        }
    }



    /// Remove the given item.
    /// Return the item if it was found and removed,
    /// None otherwise.
    pub fn remove(&mut self, key: T::Key) -> Option<T> {
        if let Some(node_index) = self.items.remove(&key) {
            let node : &mut TreeNode<T> = &mut self.tree[node_index as usize];
            if let Some(payload_index) = node.node_payload.iter().position(|x|x.get_key()==key) {
                let t = node.node_payload.swap_remove(payload_index);
                self.remove_node_if_unused(node_index as usize);
                Some(t)
            } else {
                None //We can only get here if some of the user's trait-implementations panic
            }
        } else {
            None
        }

    }

    /// Insert the given item into the QuadBTree.
    /// The performance of this operation is approximately O(1), amortized.
    pub fn insert(&mut self, item:T) {
        self.insert_impl(item, 0);
    }

    fn insert_impl(&mut self, item:T, node_index: usize) {
        let self_tree_len = self.tree.len();
        let node = &mut self.tree[node_index];

        let item_bb:BB = item.get_bb();
        let off : BB = item_bb - node.bb.top_left;
        let x1 = (off.top_left.x>>node.sub_cell_shift).clamp(0,7);
        let x2 = (off.bottom_right.x>>node.sub_cell_shift).clamp(0,7);
        let y1 = (off.top_left.y>>node.sub_cell_shift).clamp(0,7);
        let y2 = (off.bottom_right.y>>node.sub_cell_shift).clamp(0,7);

        if node.sub_cell_size >= 8 && x1==x2 && y1==y2 {
            let bitmap_place_index = x1 + (y1<<3);
            if node.node_bitmap & (1<<(bitmap_place_index as usize)) != 0 {
                // Child node exists
                let insertion_slot_index = get_set_bits_below(node.node_bitmap, bitmap_place_index as u64);
                let child_node_index = node.node_children[insertion_slot_index as usize] as usize;
                debug_assert_eq!(node.node_bitmap.count_ones() as usize, node.node_children.len());
                self.insert_impl(item, child_node_index);

            } else {
                // Child node must be created
                let insertion_slot_index = get_set_bits_below(node.node_bitmap, bitmap_place_index as u64);
                node.node_bitmap |= 1<<(bitmap_place_index as usize);

                let top_left = node.bb.top_left + Pos {
                    x: x1<<node.sub_cell_shift,
                    y: y1<<node.sub_cell_shift,
                };
                let new_bb = BB {
                    top_left,
                    bottom_right: top_left + Pos{x:node.sub_cell_size-1,y:node.sub_cell_size-1}
                };
                {
                    debug_assert_eq!(new_bb.top_left.x, node.bb.top_left.x + x1*node.sub_cell_size);
                    debug_assert_eq!(new_bb.top_left.y, node.bb.top_left.y + y1*node.sub_cell_size);
                }

                let sub_cell_size = node.sub_cell_size/8;
                let sub_cell_shift = node.sub_cell_shift-3;
                let new_node;
                let new_child_node;
                if let Some(free) = self.first_free {
                    new_child_node = free as usize;
                    node.node_children.insert(insertion_slot_index as usize, new_child_node as u32);
                    debug_assert_eq!(node.node_bitmap.count_ones() as usize, node.node_children.len());
                    new_node = &mut self.tree[free as usize];
                    self.first_free = new_node.next_free;
                    new_node.bb = new_bb;
                    new_node.sub_cell_size = sub_cell_size;
                    new_node.sub_cell_shift = sub_cell_shift;
                    debug_assert_eq!(new_node.node_bitmap,0);
                    debug_assert_eq!(new_node.node_payload.len(),0);
                    debug_assert_eq!(new_node.node_children.len(),0);
                    new_node.parent = node_index as u32;
                    new_node.node_place = bitmap_place_index as u32;
                } else {
                    new_child_node = self_tree_len;
                    node.node_children.insert(insertion_slot_index as usize, new_child_node as u32);
                    debug_assert_eq!(node.node_bitmap.count_ones() as usize, node.node_children.len());
                    self.tree.push(TreeNode {
                        bb:new_bb,
                        sub_cell_size,
                        sub_cell_shift,
                        node_bitmap: 0,
                        node_payload: smallvec![],
                        node_children: smallvec![],
                        next_free: None,
                        parent: node_index as u32,
                        node_place: bitmap_place_index as u32
                    });
                }
                assert!( (new_child_node as u64) < u32::MAX as u64);

                self.insert_impl(item, new_child_node);
            }
        } else {
            self.items.insert(item.get_key(), node_index as u32);
            node.node_payload.push(item);
        }
    }
}


#[cfg(test)]
mod tests {
    use crate::{QuadBTree, BB, TreeNodeItem};
    use rand::Rng;
    use rand::prelude::*;
    use rand_pcg::Pcg64;
    use std::collections::{HashSet};
    use test::Bencher;
    use std::hint::black_box;
    use savefile::{WithSchema, Serialize, Deserialize, Introspect};

    extern crate test;

    pub(crate) fn for_test_get_first_free<T:TreeNodeItem+WithSchema+Serialize+Deserialize+Introspect>(tree:&QuadBTree<T>) -> Option<u32> {
        tree.first_free
    }



    #[derive(Clone,Copy,PartialEq,Debug,Eq,Hash,Savefile)]
    pub struct TestItem {
        id: u32,
        pos: BB,
    }


    impl TreeNodeItem for TestItem {
        type Key = u32;

        fn get_bb(&self) -> BB {
            self.pos
        }

        fn get_key(&self) -> Self::Key {
            self.id
        }

        fn move_item(&mut self, new_bb: BB) {
            self.pos = new_bb;
        }
    }



    fn random_insert_and_query_test(seed:u64, operations: u32, deletes_and_moves: bool, gridsize: i32, max_size: Option<i32>) {

        let max_size = max_size.unwrap_or(i32::MAX);
        let mut rng = Pcg64::seed_from_u64(seed);

        let mut g = QuadBTree::new(gridsize);
        let mut all_inserted:Vec<TestItem> = Vec::new();
        for i in 0..operations {
            if deletes_and_moves && rng.gen_bool(0.33) && all_inserted.is_empty()==false {
                let to_move_index = rng.gen_range(0..all_inserted.len());
                let to_move_key = all_inserted[to_move_index].id;
                let x1 = rng.gen_range(0..gridsize);
                let y1 = rng.gen_range(0..gridsize);
                let x2 = rng.gen_range(x1..(x1.saturating_add(max_size)).min(gridsize));
                let y2 = rng.gen_range(y1..(y1.saturating_add(max_size)).min(gridsize));
                let pos = BB::new(x1,y1,x2,y2);
                assert!(g.move_item(to_move_key,pos));
                all_inserted[to_move_index].pos = pos;
            }
            if deletes_and_moves && rng.gen_bool(0.5) && all_inserted.is_empty()==false {
                let to_remove_index = rng.gen_range(0..all_inserted.len());
                let to_remove_key = all_inserted[to_remove_index].id;
                g.remove(to_remove_key);
                all_inserted.swap_remove(to_remove_index);
            } else {
                let x1 = rng.gen_range(0..gridsize);
                let y1 = rng.gen_range(0..gridsize);
                let x2 = rng.gen_range(x1..(x1.saturating_add(max_size)).min(gridsize));
                let y2 = rng.gen_range(y1..(y1.saturating_add(max_size)).min(gridsize));
                let test_item = TestItem {
                    id: i,
                    pos: BB::new(x1,y1,x2,y2)
                };
                //println!("Inserting {:?}", test_item);

                g.insert(test_item);
                all_inserted.push(test_item);
            }
        }
        //println!("After {} operations, has {} tree nodes.", operations, g.tree.len());
        for _i in 0..10 {
            let x1 = rng.gen_range(0..gridsize);
            let y1 = rng.gen_range(0..gridsize);
            let x2 = rng.gen_range(x1..gridsize);
            let y2 = rng.gen_range(y1..gridsize);
            let pos= BB::new(x1,y1,x2,y2);
            //println!("Querying {:?}", pos);
            let g_hit :HashSet<TestItem>= g.query(pos).iter().cloned().cloned().collect();
            let correct_hits:HashSet<TestItem> = all_inserted.iter().filter(|x|x.pos.overlaps(pos)).cloned().collect();
            assert_eq!(g_hit,correct_hits);
        }

        let mut overlaps = HashSet::new();
        g.find_all_overlapping_neighbors(|a,b|{overlaps.insert((a.id, b.id));});
        for item_a in all_inserted.iter() {
            for item_b in all_inserted.iter() {
                if item_a.id==item_b.id {
                    continue;
                }
                let does_overlap = item_a.pos.overlaps(item_b.pos);
                let expected_overlaps = if does_overlap {1} else {0};
                let mut found_overlaps = 0;
                if overlaps.contains(&(item_a.id,item_b.id)) {
                    found_overlaps += 1;
                }
                if overlaps.contains(&(item_b.id,item_a.id)) {
                    found_overlaps += 1;
                }
                assert_eq!(expected_overlaps, found_overlaps);
            }
        }

        overlaps.clear();
        g.find_all_overlapping_neighbors2(&g, |a,b|{
            assert!(overlaps.insert((a.id, b.id)));
        });
        for item_a in all_inserted.iter() {
            for item_b in all_inserted.iter() {
                let does_overlap = item_a.pos.overlaps(item_b.pos);
                let expected_overlaps = if does_overlap {1} else {0};
                let mut found_overlaps = 0;
                if overlaps.contains(&(item_a.id,item_b.id)) {
                    found_overlaps += 1;
                }
                assert_eq!(expected_overlaps, found_overlaps);
            }
        }

        for item in all_inserted {
            let t = g.get_by_key(item.get_key()).unwrap();
            assert_eq!(&item, t);
            let t = g.remove(item.get_key()).unwrap();
            assert_eq!(item, t);
        }




    }
    #[test]
    fn exhaustive_insert_query_single_insert_test() {
        for seed in 0..1000000 {
            //println!("Testing seed {}", seed);
            random_insert_and_query_test(seed,1,false,1024, None);
        }
    }
    #[test]
    fn exhaustive_insert_remove_query_single_op_test() {
        for seed in 0..100000 {
            println!("Testing seed {}", seed);
            random_insert_and_query_test(seed,2,true,1024, None);
        }
    }
    #[test]
    fn exhaustive_insert_remove_query_test() {
        for seed in 0..100000 {
            //println!("Testing seed {}", seed);
            random_insert_and_query_test(seed,100,true,1024, None);
        }
    }
    #[test]
    fn exhaustive_insert_query_test() {
        for seed in 0..100000 {
            //println!("Testing seed {}", seed);
            random_insert_and_query_test(seed,10,false,1024, None);
        }
    }
    #[test]
    fn exhaustive_insert_query_test_large_grid() {
        for seed in 0..100000 {
            //println!("Testing seed {}", seed);

            random_insert_and_query_test(seed,50,true,65536*1024, Some(512));
        }
    }
    #[test]
    fn exhaustive_insert_test_seed() {

        random_insert_and_query_test(190,2,true,1024, None);
    }
    #[test]
    fn exhaustive_insert_remove_test_seed9() {
        random_insert_and_query_test(9,2,true,1024, None);
    }
    #[test]
    fn basic_remove_test1() {
        let mut g = QuadBTree::new(256);
        let test_item = TestItem {
            id: 42,
            pos: BB::new(10, 10, 10, 10)
        };
        g.insert(test_item);
        assert_eq!(g.query(BB::new(9,9,11,11)), vec![&test_item]);
        g.remove(42);

        assert_eq!(g.query(BB::new(9,9,11,11)), Vec::<&TestItem>::new());

        let first_free = for_test_get_first_free(&g);
        assert_eq!(first_free, Some(1));
    }
    #[test]
    fn basic_insert_test1() {
        let mut q = QuadBTree::new(256);
        assert_eq!(q.query(BB::new(0,0,255,255)), Vec::<&TestItem>::new());

        let test_item = TestItem {
            id: 42,
            pos: BB::new(10,10,15,15)
        };
        q.insert(test_item);

        assert_eq!(q.query(BB::new(9,9,11,11)), vec![&test_item]);

        assert_eq!(q.query(BB::new(9,9,16,16)), vec![&test_item]);
        assert_eq!(q.query(BB::new(2,2,4,4)), Vec::<&TestItem>::new());
        q.remove(42);
        assert_eq!(q.query(BB::new(9,9,16,16)), Vec::<&TestItem>::new());

        let _ = q.clone(); //Just check clone works
    }
    #[test]
    fn basic_find_neighbors_test() {
        let mut q = QuadBTree::new(256);
        let test_item1 = TestItem {
            id: 42,
            pos: BB::new(10, 10, 15, 15)
        };
        let test_item2 = TestItem {
            id: 43,
            pos: BB::new(20, 20, 25, 25)
        };
        let test_item3 = TestItem {
            id: 44,
            pos: BB::new(25, 25, 27, 27)
        };
        q.insert(test_item1);
        q.insert(test_item2);
        q.insert(test_item3);
        let mut found = Vec::new();
        q.find_all_overlapping_neighbors(|a,b|{
            found.push((a.id,b.id));
           println!("Found neighbors: {:?} & {:?}", a.id, b.id);
        });
        assert_eq!(found,vec![(44,43)]);
    }
    #[test]
    fn basic_find_neighbors2_test() {
        let mut q_a = QuadBTree::new(256);
        let mut q_b = QuadBTree::new(256);
        let test_item1a = TestItem {
            id: 42,
            pos: BB::new(10, 10, 25, 25)
        };
        let test_item2b = TestItem {
            id: 43,
            pos: BB::new(20, 20, 25, 25)
        };
        let test_item3b = TestItem {
            id: 44,
            pos: BB::new(26, 26, 27, 27)
        };
        q_a.insert(test_item1a);
        q_b.insert(test_item2b);
        q_b.insert(test_item3b);
        let mut found = Vec::new();
        q_a.find_all_overlapping_neighbors2(&q_b, |a,b|{
            println!("Found neighbors: {:?} & {:?}", a.id, b.id);
            found.push((a.id,b.id));
        });
        assert_eq!(found,vec![(42,43)]);
    }

    #[test]
    fn bb_test1() {
        let bb1 = BB::new(20, 20, 25, 25);
        let bb2 = BB::new(21, 21, 25, 25);
        assert_eq!(bb1.distance_to(bb2), 0);
        assert!(bb1.overlaps(bb2));
    }
    #[bench]
    fn benchmark_random_queries(b: &mut Bencher) {
        let mut rng = Pcg64::seed_from_u64(42);
        let gridsize = 8192;
        let max_size = 4;
        let max_query_size = 32;

        let mut g = QuadBTree::new(gridsize);
        for i in 0..4_000 {
            let x1 = rng.gen_range(0..gridsize);
            let y1 = rng.gen_range(0..gridsize);
            let x2 = rng.gen_range(x1..(x1+max_size).min(gridsize));
            let y2 = rng.gen_range(y1..(y1+max_size).min(gridsize));
            let test_item = TestItem {
                id: i,
                pos: BB::new(x1,y1,x2,y2)
            };
            g.insert(test_item);
        }
        println!("Tree nodes: {}",g.tree.len());

        let mut rng = thread_rng();
        {
            let x1 = rng.gen_range(0..gridsize);
            let y1 = rng.gen_range(0..gridsize);
            let x2 = rng.gen_range(x1..(x1+max_query_size).min(gridsize));
            let y2 = rng.gen_range(y1..(y1+max_query_size).min(gridsize));

            let pos = BB::new(x1,y1,x2,y2);
            b.iter(||{
                g.query_fn(pos, |item|{
                    black_box(item);
                });
            });
        }
    }
    #[bench]
    fn benchmark_find_neighbors(b: &mut Bencher) {
        let mut rng = Pcg64::seed_from_u64(42);
        let gridsize = 8192;
        let max_size = 64;


        let mut g = QuadBTree::new(gridsize);
        for i in 0..1_000 {
            let x1 = rng.gen_range(0..gridsize);
            let y1 = rng.gen_range(0..gridsize);
            let x2 = rng.gen_range(x1..(x1+max_size).min(gridsize));
            let y2 = rng.gen_range(y1..(y1+max_size).min(gridsize));
            let test_item = TestItem {
                id: i,
                pos: BB::new(x1,y1,x2,y2)
            };
            g.insert(test_item);
        }
        println!("Tree nodes: {}, top payloads: {}",g.tree.len(), g.tree[0].node_payload.len());

        {

            b.iter(||{
                let mut count = 0;
                g.find_all_overlapping_neighbors(|a,b|{
                    black_box(a);
                    black_box(b);
                    count+=1;
                });
                println!("Count: {}",count);

            });
        }
    }
    #[test]
    fn benchmark_find_neighbors_test() {
        let mut rng = Pcg64::seed_from_u64(42);
        let gridsize = 8192;
        let max_size = 2;

        let mut g = QuadBTree::new(gridsize);
        for i in 0..1_000 {
            let x1 = rng.gen_range(0..gridsize);
            let y1 = rng.gen_range(0..gridsize);
            let x2 = rng.gen_range(x1..(x1+max_size).min(gridsize));
            let y2 = rng.gen_range(y1..(y1+max_size).min(gridsize));
            let test_item = TestItem {
                id: i,
                pos: BB::new(x1,y1,x2,y2)
            };
            g.insert(test_item);
        }
        println!("Tree nodes: {}, top payloads: {}",g.tree.len(), g.tree[0].node_payload.len());

        {

            {

                g.find_all_overlapping_neighbors(|a,b|{
                    black_box(a);
                    black_box(b);

                });
            };
        }
    }
}
