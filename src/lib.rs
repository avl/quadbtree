#![feature(int_log)]

use std::ops::Sub;
use std::collections::HashMap;

#[derive(Clone,Copy,PartialEq)]
pub struct Pos {
    pub x: i32,
    pub y: i32,
}

/// Bounding box for an item. The bounding box includes
/// both the top_left corner and the bottom right corner.
/// An object within a bounding box is contained entirely within
/// the box.
#[derive(Clone,Copy,PartialEq)]
pub struct BB {
    pub top_left: Pos,
    pub bottom_right: Pos,
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

impl Sub<Pos> for BB {
    type Output = BB;

    fn sub(self, rhs: Pos) -> Self::Output {
        BB {
            top_left: self.top_left - rhs,
            bottom_right: self.bottom_right - rhs,
        }
    }
}

trait TreeNodeItem {
    /// The type of a unique key identifying a specific item.
    /// If you don't need this, you can use the BB as the key,
    /// or even Self.
    type Key : PartialEq;

    /// Return bounding box for the given item
    fn get_bb(&self) -> BB;

    /// Return a key identifying a particular item.
    fn get_key(&self) -> Self::Key;

}

fn get_set_bits_below(bitmap:u64, bit_index: u64) -> u32 {
    let below_mask = ((1<<bit_index) - 1);
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
    parent: u32, //0 means no parent,
    node_place: u32,
}

pub struct QuadBTree<T> {
    tree: Vec<TreeNode<T>>, //The first position is always the root
    first_free: Option<u32>, //If there are free nodes, this is the first free
    items: HashMap<T::Key, u32>, //map item key to tree node
}

impl QuadBTree<T> {

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
            next_free: None
        };
        QuadBTree {
            tree: vec![root],
            first_free: None
        }
    }

    fn remove_node_if_unused(&mut self, node_index: usize) {
        let node : &mut TreeNode<T> = &mut self.tree[node_index];
        let parent_index = node.parent;
        if node.node_payload.is_empty() == false || node.node_children.is_empty() {
            return;
        }


        if parent_index != 0 {
            let parent_node:&mut TreeNode<T> = &mut self.tree[parent_index];
            let place = get_set_bits_below(parent_node.node_bitmap, node.node_place as u64);
            parent_node.node_children.remove(place as usize);
            parent_node.node_bitmap &= !(1u64<<place);
            self.remove_node_if_unused(parent_index as usize)
        }
    }
compile_error!("Finish!")
    /// Remove the given item.
    /// Return true if the item was found and removed,
    /// false otherwise.
    pub fn remove(&mut self, key: T::Key) -> bool {
        if let Some(index) = self.items.remove(key) {
            let node : &mut TreeNode<T> = &mut self.tree[index];
            if let Some(index) = node.node_payload.iter().position(|x|x==key) {
                node.node_payload.swap_remove(index);

                self.remove_node_if_unused(index);

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
        let node = &mut self.tree[node_index];

        let item_bb:BB = item.get_bb();
        let off : BB = item_bb - node.bb.top_left;
        let x1 = off.top_left.x>>node.sub_cell_shift;
        let x2 = off.bottom_right.x>>node.sub_cell_shift;
        let y1 = off.top_left.y>>node.sub_cell_shift;
        let y2 = off.bottom_right.y>>node.sub_cell_shift;

        if x1==x2 && y1==y2 && node.sub_cell_size >= 8 {
            let bitmap_index = x1 + y1<<3;
            if node.node_bitmap & (1<<(bitmap_index as usize)) != 0 {
                // Child node exists
                let insertion_place = get_set_bits_below(node.node_bitmap, bitmap_index as u64);
                self.insert_impl(item, node.node_children[insertion_place as usize] as usize);

            } else {
                // Child node must be created
                let insertion_place = get_set_bits_below(node.node_bitmap, bitmap_index as u64);
                node.node_bitmap |= (1<<(bitmap_index as usize));

                let new_node;
                let new_child_node;
                if let Some(free) = self.first_free {
                    new_child_node = free as usize;
                    new_node = &mut self.tree[free as usize];
                    self.first_free = new_node.next_free;
                } else {
                    new_child_node = self.tree.len();
                    let top_left = Pos {
                        x: x1<<node.sub_cell_shift,
                        y: y1<<node.sub_cell_shift,
                    };
                    self.tree.push(TreeNode {
                        bb: BB {
                            top_left,
                            bottom_right: top_left + Pos{x:node.sub_cell_size-1,y:node.sub_cell_size-1}
                        },
                        sub_cell_size: node.sub_cell_size/8,
                        sub_cell_shift: node.sub_cell_shift-3,
                        node_bitmap: 0,
                        node_payload: vec![],
                        node_children: vec![],
                        next_free: None
                    });
                }
                assert!(new_child_node as u64 < u32::MAX as u64);
                node.node_children.insert(insertion_place as usize, new_child_node as u32);

                self.insert_impl(item, new_child_node);
            }
        } else {
            node.node_payload.push(item);
            self.items.insert(item.get_key(), node_index as u32);
        }
    }
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
