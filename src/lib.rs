#![feature(int_log)]
#![feature(stdsimd)]
#![feature(bench_black_box)]
#![feature(test)]

use std::ops::{Sub, Add};
use std::collections::HashMap;
use std::hash::Hash;
use std::arch::x86_64::{__m128i, _mm_set1_epi8, _mm_cvtsi128_si64x, _mm_and_si128, _mm_set1_epi64x};

#[cfg(all(target_feature = "avx512vl", target_feature="avx512bw"))]
use std::arch::x86_64::{_mm_broadcastb_epi8, _mm_maskz_broadcastb_epi8};

#[derive(Clone,Copy,PartialEq,Debug,Eq,Hash)]
pub struct Pos {
    pub x: i32,
    pub y: i32,
}

/// Bounding box for an item. The bounding box includes
/// both the top_left corner and the bottom right corner.
/// An object within a bounding box is contained entirely within
/// the box.
#[derive(Clone,Copy,PartialEq,Debug,Eq,Hash)]
pub struct BB {
    pub top_left: Pos,
    pub bottom_right: Pos,
}

impl BB {
    /// Create a new bounding box from the given coordinates.
    /// x1,y1 must be the top left corner, and x2,y2 must be the
    /// lower right.
    pub fn new(x1:i32, y1:i32, x2:i32, y2:i32) -> BB {
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
    #[inline(always)]
    pub fn overlaps(self, other: BB) -> bool {
        if  self.top_left.x >= other.bottom_right.x ||
            self.top_left.y >= other.bottom_right.y ||
            self.bottom_right.x <= other.top_left.x ||
            self.bottom_right.y <= other.top_left.y {
            false
        } else {
            true
        }
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

pub trait TreeNodeItem {
    /// The type of a unique key identifying a specific item.
    /// If you don't need this, you can use the BB as the key,
    /// or even Self.
    type Key : PartialEq+Eq+Hash;

    /// Return bounding box for the given item
    fn get_bb(&self) -> BB;

    /// Return a key identifying a particular item.
    fn get_key(&self) -> Self::Key;

}

fn get_set_bits_below(bitmap:u64, bit_index: u64) -> u32 {
    let below_mask = (1<<bit_index) - 1;
    (bitmap & below_mask).count_ones()
}

struct TreeNode<T> {
    bb: BB,
    sub_cell_size: i32,
    sub_cell_shift: u32,
    node_bitmap: u64,
    node_payload: Vec<T>,
    node_children: Vec<u32>,
    next_free: Option<u32>, //If TreeNode is part of free-list, this is the next free
    parent: u32, //The root has parent id 0. It doesn't really have a parent of its own, but it has 0 here.
    node_place: u32,
}

pub struct QuadBTree<T:TreeNodeItem> {
    tree: Vec<TreeNode<T>>, //The first position is always the root
    first_free: Option<u32>, //If there are free nodes, this is the first free
    items: HashMap<T::Key, u32>, //map item key to tree node
}



impl<T:TreeNodeItem> QuadBTree<T> {

    /// Create a new QuadBTree of the given size.
    /// Only coordinates in the range 0..size are allowed
    /// for objects which are to be put into the tree.
    /// This means that if size is 256, the coordinate (256,0) is _not_valid, but
    /// (255,0) is.
    pub fn new(size: i32) -> QuadBTree<T> {
        let shift = (size as u32).log2();
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
            node_payload: vec![],
            node_children: vec![],
            next_free: None,
            parent: 0,
            node_place: 0
        };
        QuadBTree {
            tree: vec![root],
            first_free: None,
            items: HashMap::new()
        }
    }

    fn remove_node_if_unused(&mut self, node_index: usize) {
        if node_index == 0 {
            return; //Root node is never removed.
        }
        let node : &mut TreeNode<T> = &mut self.tree[node_index];
        let parent_index = node.parent as usize;
        if node.node_payload.is_empty() == false || node.node_children.is_empty()==false {
            return;
        }


        debug_assert_eq!(node.node_bitmap,0);
        node.next_free = self.first_free;
        self.first_free = Some(node_index as u32);
        {
            let node_node_place = node.node_place;
            let parent_node:&mut TreeNode<T> = &mut self.tree[parent_index];
            let place_index = get_set_bits_below(parent_node.node_bitmap, node_node_place as u64);
            parent_node.node_children.remove(place_index as usize);
            parent_node.node_bitmap &= !(1u64<<node_node_place);
            debug_assert_eq!(parent_node.node_bitmap.count_ones() as usize, parent_node.node_children.len());
            self.remove_node_if_unused(parent_index)
        }
    }

    #[cfg(all(target_feature = "avx512vl", target_feature="avx512bw"))]
    #[inline(always)]
    fn spread_op(x1:i32,y1:i32,x2:i32, y2:i32) -> u64 {
        let each_row = ((2<<(x2 as u8))-1) & ! ((1<<x1)-1);
        let rows_mask = ((2<<(y2 as u8))-1) & ! ((1<<y1)-1);
        unsafe {
            let xtemp =  _mm_set1_epi8(each_row);
            let x = _mm_maskz_broadcastb_epi8(rows_mask, xtemp);
            let x = _mm_cvtsi128_si64x(x) as u64;
            x
        }
    }

    #[cfg(not(all(target_feature = "avx512vl", target_feature="avx512bw")))]
    #[inline(always)]
    fn spread_op(x1:i32,y1:i32,x2:i32, y2:i32) -> u64 {
        let each_row = ((2u8<<(x2 as u8)).wrapping_sub(1u8)) & ! ((1<<x1)-1);
        let rows_mask = ((256u64<<(y2 as u64*8)).wrapping_sub(1)) & ! ((1u64<<(y1*8))-1);
        unsafe {
            let mm_all_rows = _mm_set1_epi8(each_row as i8);
            let mm_rows_mask: __m128i = _mm_set1_epi64x(rows_mask as i64);
            let result = _mm_and_si128(mm_all_rows, mm_rows_mask);
            let x = _mm_cvtsi128_si64x(result) as u64;
            return x;
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

        let off : BB = bb - node.bb.top_left;
        let x1 = (off.top_left.x>>node.sub_cell_shift).clamp(0,7);
        let x2 = (off.bottom_right.x>>node.sub_cell_shift).clamp(0,7);
        let y1 = (off.top_left.y>>node.sub_cell_shift).clamp(0,7);
        let y2 = (off.bottom_right.y>>node.sub_cell_shift).clamp(0,7);

        let bb_mask = Self::spread_op(x1,y1,x2,y2);
        let mut child_mask:u64 = node.node_bitmap&bb_mask;
        while child_mask!=0 {
            let cur_child = child_mask.trailing_zeros() as u64;
            let cur_child_mask = 1<<cur_child;
            child_mask &= !cur_child_mask;
            let child_place = ((cur_child_mask-1)&node.node_bitmap).count_ones();
            let child_node_index = node.node_children[child_place as usize];
            self.query_impl(child_node_index as usize, bb, &mut *f);
        }
    }
    #[inline(always)]
    pub fn query(&self, bb:BB) -> Vec<&T> {
        let mut ret = vec![];
        self.query_impl(0, bb, &mut |item|ret.push(item));
        ret
    }
    #[inline(always)]
    pub fn query_fn<F:FnMut(&T)>(&self, bb: BB, mut f:F) {
        self.query_impl(0,bb,&mut f)
    }


    /// Remove the given item.
    /// Return true if the item was found and removed,
    /// false otherwise.
    pub fn remove(&mut self, key: T::Key) -> bool {
        if let Some(node_index) = self.items.remove(&key) {
            let node : &mut TreeNode<T> = &mut self.tree[node_index as usize];
            if let Some(placement_index) = node.node_payload.iter().position(|x|x.get_key()==key) {
                node.node_payload.swap_remove(placement_index);
                self.remove_node_if_unused(node_index as usize);
                true
            } else {
                false //We can only get here if some of the user's trait-implementations panic
            }
        } else {
            false
        }

    }

    /// Insert the given item into the QuadBTree
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

        if x1==x2 && y1==y2 && node.sub_cell_size >= 8 {
            let bitmap_index = x1 + (y1<<3);
            if node.node_bitmap & (1<<(bitmap_index as usize)) != 0 {
                // Child node exists
                let insertion_place = get_set_bits_below(node.node_bitmap, bitmap_index as u64);
                let child_node_index = node.node_children[insertion_place as usize] as usize;
                debug_assert_eq!(node.node_bitmap.count_ones() as usize, node.node_children.len());
                self.insert_impl(item, child_node_index);

            } else {
                // Child node must be created
                let insertion_place = get_set_bits_below(node.node_bitmap, bitmap_index as u64);
                node.node_bitmap |= 1<<(bitmap_index as usize);

                let top_left = Pos {
                    x: x1<<node.sub_cell_shift,
                    y: y1<<node.sub_cell_shift,
                };                let bb = BB {
                    top_left,
                    bottom_right: top_left + Pos{x:node.sub_cell_size-1,y:node.sub_cell_size-1}
                };
                let sub_cell_size = node.sub_cell_size/8;
                let sub_cell_shift = node.sub_cell_shift-3;
                let new_node;
                let new_child_node;
                if let Some(free) = self.first_free {
                    new_child_node = free as usize;
                    node.node_children.insert(insertion_place as usize, new_child_node as u32);
                    debug_assert_eq!(node.node_bitmap.count_ones() as usize, node.node_children.len());
                    new_node = &mut self.tree[free as usize];
                    self.first_free = new_node.next_free;
                    new_node.bb = bb;
                    new_node.sub_cell_size = sub_cell_size;
                    new_node.sub_cell_shift = sub_cell_shift;
                    debug_assert_eq!(new_node.node_bitmap,0);
                    debug_assert_eq!(new_node.node_payload.len(),0);
                    debug_assert_eq!(new_node.node_children.len(),0);
                    new_node.parent = node_index as u32;
                    new_node.node_place = bitmap_index as u32;
                } else {
                    new_child_node = self_tree_len;
                    node.node_children.insert(insertion_place as usize, new_child_node as u32);
                    debug_assert_eq!(node.node_bitmap.count_ones() as usize, node.node_children.len());
                    self.tree.push(TreeNode {
                        bb,
                        sub_cell_size,
                        sub_cell_shift,
                        node_bitmap: 0,
                        node_payload: vec![],
                        node_children: vec![],
                        next_free: None,
                        parent: node_index as u32,
                        node_place: bitmap_index as u32
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
    use std::collections::HashSet;
    use test::Bencher;
    use std::hint::black_box;
    use std::time::Instant;

    extern crate test;

    pub(crate) fn for_test_get_first_free<T:TreeNodeItem>(tree:&QuadBTree<T>) -> Option<u32> {
        tree.first_free
    }


    #[derive(Clone,Copy,PartialEq,Debug,Eq,Hash)]
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
    }



    fn random_insert_and_query_test(seed:u64, operations: u32, deletes: bool, gridsize: i32, max_size: Option<i32>) {

        let max_size = max_size.unwrap_or(i32::MAX);
        let mut rng = Pcg64::seed_from_u64(seed);

        let mut g = QuadBTree::new(gridsize);
        let mut all_inserted:Vec<TestItem> = Vec::new();
        for i in 0..operations {
            if deletes && rng.gen_bool(0.5) && all_inserted.is_empty()==false {
                let to_remove_index = rng.gen_range(0..all_inserted.len());
                let to_remove_key = all_inserted[to_remove_index].id;
                g.remove(to_remove_key);
                all_inserted.swap_remove(to_remove_index);
            } else {
                let x1 = rng.gen_range(0..gridsize);
                let y1 = rng.gen_range(0..gridsize);
                let x2 = rng.gen_range(x1..(x1+max_size).min(gridsize));
                let y2 = rng.gen_range(y1..(y1+max_size).min(gridsize));
                let test_item = TestItem {
                    id: i,
                    pos: BB::new(x1,y1,x2,y2)
                };
                //println!("Inserting {:?}", test_item);

                g.insert(test_item);
                all_inserted.push(test_item);
            }
        }
        println!("After {} operations, has {} tree nodes.", operations, g.tree.len());
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


    }
    #[test]
    fn exhaustive_insert_query_single_insert_test() {
        for seed in 0..100000 {
            //println!("Testing seed {}", seed);
            random_insert_and_query_test(seed,1,false,1024, None);
        }
    }
    #[test]
    fn exhaustive_insert_remove_query_single_op_test() {
        for seed in 0..100000 {
            //println!("Testing seed {}", seed);
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
    fn exhaustive_insert_test_seed118() {
        random_insert_and_query_test(118,10,false,1024, None);
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
    }

    #[bench]
    fn benchmark_random_queries(b: &mut Bencher) {
        let mut rng = Pcg64::seed_from_u64(42);
        let gridsize = 8192;
        let max_size = 1024;
        compile_error!("Figure out why this isn't faster!");
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
        println!("Tree nodes: {}",g.tree.len());

        let mut rng = thread_rng();
        {
            let x1 = rng.gen_range(0..gridsize);
            let y1 = rng.gen_range(0..gridsize);
            let x2 = rng.gen_range(x1..(x1+max_size).min(gridsize));
            let y2 = rng.gen_range(y1..(y1+max_size).min(gridsize));

            let pos = BB::new(x1,y1,x2,y2);
            b.iter(||{
                g.query_fn(pos, |item|{
                    black_box(item);
                });
            });

        }

    }
}
