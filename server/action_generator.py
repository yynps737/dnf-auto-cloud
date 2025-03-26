#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
动作生成模块，根据图像识别结果生成游戏操作指令
优化版 - 更智能的决策逻辑和状态机实现
"""

import logging
import random
import time
import math
import numpy as np
from collections import deque

from config.settings import DNF, BEHAVIOR

logger = logging.getLogger("DNFAutoCloud")

# 历史决策保存
decision_history = deque(maxlen=10)
state_transitions = {}

class GameState:
    """游戏状态枚举"""
    UNKNOWN = "unknown"
    LOADING = "loading"
    LOBBY = "lobby"
    DUNGEON_SELECTION = "dungeon_selection"
    ROOM_SELECTION = "room_selection"
    IN_DUNGEON = "in_dungeon"
    IN_BATTLE = "in_battle"
    REST = "rest"
    PICKUP_ITEMS = "pickup_items"
    TALKING = "talking"
    INVENTORY = "inventory"
    DEAD = "dead"

class ActionPriority:
    """动作优先级"""
    CRITICAL = 0.5       # 紧急动作（如低血量使用药水）
    HIGH = 0.8           # 高优先级（如打BOSS）
    NORMAL = 1.0         # 正常优先级
    LOW = 1.5            # 低优先级
    BACKGROUND = 2.0     # 后台任务

def generate_actions(detections, game_state):
    """
    根据检测结果和游戏状态生成动作 - 优化版
    
    参数:
        detections (list): 检测结果列表
        game_state (dict): 游戏状态信息
        
    返回:
        list: 操作指令列表
    """
    actions = []
    
    try:
        # 推断当前游戏状态
        current_state = infer_game_state(detections, game_state)
        
        # 记录状态转换
        record_state_transition(game_state.get("previous_state", GameState.UNKNOWN), current_state)
        
        # 检查玩家状态是否需要紧急处理
        emergency_actions = check_emergency(detections, game_state)
        if emergency_actions:
            actions.extend(emergency_actions)
            # 在紧急情况下，可能需要提前返回而不执行其他操作
            if any(a.get("critical", False) for a in emergency_actions):
                return actions
        
        # 根据当前状态生成动作
        if current_state == GameState.IN_BATTLE:
            battle_actions = generate_battle_actions(detections, game_state)
            actions.extend(battle_actions)
        
        elif current_state == GameState.IN_DUNGEON:
            exploration_actions = generate_exploration_actions(detections, game_state)
            actions.extend(exploration_actions)
        
        elif current_state == GameState.PICKUP_ITEMS:
            pickup_actions = generate_pickup_actions(detections, game_state)
            actions.extend(pickup_actions)
        
        elif current_state == GameState.TALKING:
            talking_actions = generate_talking_actions(detections, game_state)
            actions.extend(talking_actions)
        
        elif current_state == GameState.LOBBY:
            lobby_actions = generate_lobby_actions(detections, game_state)
            actions.extend(lobby_actions)
        
        elif current_state == GameState.DUNGEON_SELECTION:
            selection_actions = generate_dungeon_selection_actions(detections, game_state)
            actions.extend(selection_actions)
        
        elif current_state == GameState.ROOM_SELECTION:
            room_actions = generate_room_selection_actions(detections, game_state)
            actions.extend(room_actions)
        
        elif current_state == GameState.DEAD:
            dead_actions = generate_dead_actions(detections, game_state)
            actions.extend(dead_actions)
        
        else:
            # 默认动作
            default_actions = generate_default_actions(detections, game_state)
            actions.extend(default_actions)
        
        # 添加人类行为特征
        humanize_actions(actions)
        
        # 记录决策历史
        record_decision(current_state, detections, actions)
        
        # 更新游戏状态
        game_state["previous_state"] = current_state
        
        return actions
        
    except Exception as e:
        logger.error(f"生成动作时出错: {e}")
        # 出错时返回安全的默认动作（如停止移动）
        return [{"type": "stop", "reason": "error_recovery", "execution_priority": ActionPriority.CRITICAL}]

def infer_game_state(detections, game_state):
    """
    根据检测结果推断当前游戏状态
    
    参数:
        detections (list): 检测结果
        game_state (dict): 游戏状态
        
    返回:
        str: 游戏状态
    """
    # 检查是否有怪物或Boss（战斗状态）
    monsters = [d for d in detections if d["class_name"] in ["monster", "boss"]]
    if monsters:
        return GameState.IN_BATTLE
    
    # 检查是否有物品（拾取状态）
    items = [d for d in detections if d["class_name"] == "item"]
    if items and not game_state.get("inventory_full", False):
        return GameState.PICKUP_ITEMS
    
    # 检查是否有对话框（对话状态）
    dialog_boxes = [d for d in detections if d["class_name"] == "dialog"]
    if dialog_boxes:
        return GameState.TALKING
    
    # 检查是否有门（地下城探索状态）
    doors = [d for d in detections if d["class_name"] == "door"]
    if doors:
        return GameState.IN_DUNGEON
    
    # 检查是否有NPC（可能在城镇/大厅）
    npcs = [d for d in detections if d["class_name"] == "npc"]
    if npcs:
        # 区分大厅和地下城中的NPC
        if game_state.get("current_map", "").startswith("town_"):
            return GameState.LOBBY
        else:
            return GameState.IN_DUNGEON
    
    # 检查是否有地下城选择界面
    dungeon_ui = [d for d in detections if d["class_name"] == "dungeon_select"]
    if dungeon_ui:
        return GameState.DUNGEON_SELECTION
    
    # 检查是否有房间选择界面
    room_ui = [d for d in detections if d["class_name"] == "room_select"]
    if room_ui:
        return GameState.ROOM_SELECTION
    
    # 检查是否死亡
    death_ui = [d for d in detections if d["class_name"] == "death_ui"]
    if death_ui:
        return GameState.DEAD
    
    # 其他情况，根据当前位置判断
    current_map = game_state.get("current_map", "")
    if current_map.startswith("town_"):
        return GameState.LOBBY
    elif current_map:
        return GameState.IN_DUNGEON
    
    # 保持之前的状态（如果有）
    previous_state = game_state.get("previous_state")
    if previous_state:
        return previous_state
    
    # 默认为未知状态
    return GameState.UNKNOWN

def check_emergency(detections, game_state):
    """
    检查紧急情况并生成相应动作
    
    参数:
        detections (list): 检测结果
        game_state (dict): 游戏状态
        
    返回:
        list: 紧急操作指令
    """
    actions = []
    
    # 检查血量
    hp_bars = [d for d in detections if d["class_name"] == "hp_bar"]
    if hp_bars:
        hp_percent = hp_bars[0].get("percent", 100)
        
        # 血量过低时使用药水
        if hp_percent < 30:
            actions.append({
                "type": "use_item",
                "key": "f1",  # 假设F1是HP药水快捷键
                "purpose": "use_hp_potion",
                "execution_priority": ActionPriority.CRITICAL,
                "critical": True,
                "reason": f"低血量 ({hp_percent:.1f}%)"
            })
    
    # 检查蓝量
    mp_bars = [d for d in detections if d["class_name"] == "mp_bar"]
    if mp_bars:
        mp_percent = mp_bars[0].get("percent", 100)
        
        # 蓝量过低时使用药水
        if mp_percent < 20:
            actions.append({
                "type": "use_item",
                "key": "f2",  # 假设F2是MP药水快捷键
                "purpose": "use_mp_potion",
                "execution_priority": ActionPriority.HIGH,
                "reason": f"低蓝量 ({mp_percent:.1f}%)"
            })
    
    # 检查是否需要复活或返回城镇
    if game_state.get("previous_state") == GameState.DEAD:
        actions.append({
            "type": "press_key_combo",
            "keys": ["alt", "r"],  # 假设Alt+R是复活快捷键
            "purpose": "revive",
            "execution_priority": ActionPriority.CRITICAL,
            "critical": True,
            "reason": "角色已死亡，需要复活"
        })
    
    return actions

def generate_battle_actions(detections, game_state):
    """
    生成战斗状态下的操作 - 优化版
    
    参数:
        detections (list): 检测结果
        game_state (dict): 游戏状态
        
    返回:
        list: 操作指令
    """
    actions = []
    
    # 获取所有怪物
    monsters = [d for d in detections if d["class_name"] in ["monster", "boss"]]
    
    if monsters:
        # 按照优先级排序（Boss > 精英怪 > 普通怪物）
        def monster_priority(m):
            if m["class_name"] == "boss":
                return 0
            elif m.get("is_elite", False):
                return 1
            else:
                return 2
        
        monsters.sort(key=monster_priority)
        
        # 选取目标
        target = monsters[0]
        
        # 目标距离
        target_center = target["center"]
        
        # 选择合适的技能
        skill_key, skill_type = choose_skill(game_state, detections, monsters)
        
        if skill_type == "ranged":
            # 远程技能 - 站在适当距离释放
            # 首先移动到合适的位置
            ideal_distance = 200  # 理想距离（像素）
            current_pos = [game_state.get("player_x", 0), game_state.get("player_y", 0)]
            
            # 计算当前到目标的向量
            vec_to_target = [
                target_center[0] - current_pos[0],
                target_center[1] - current_pos[1]
            ]
            
            # 计算距离
            distance = math.sqrt(vec_to_target[0]**2 + vec_to_target[1]**2)
            
            # 如果太近，向后移动
            if distance < ideal_distance * 0.8:
                # 标准化向量并反向
                norm = math.sqrt(vec_to_target[0]**2 + vec_to_target[1]**2)
                if norm > 0:
                    vec_direction = [-vec_to_target[0]/norm, -vec_to_target[1]/norm]
                    back_pos = [
                        current_pos[0] + vec_direction[0] * ideal_distance * 0.2,
                        current_pos[1] + vec_direction[1] * ideal_distance * 0.2
                    ]
                    
                    actions.append({
                        "type": "move_to",
                        "position": back_pos,
                        "purpose": "adjust_range",
                        "execution_priority": ActionPriority.HIGH
                    })
            
            # 添加使用技能动作
            actions.append({
                "type": "use_skill",
                "key": skill_key,
                "target_position": target_center,
                "target_id": target.get("id", "unknown"),
                "target_type": target["class_name"],
                "skill_type": "ranged",
                "execution_priority": ActionPriority.NORMAL
            })
            
        elif skill_type == "aoe":
            # AOE技能 - 瞄准多个怪物的中心
            if len(monsters) > 1:
                # 计算多个怪物的中心点
                center_x = sum(m["center"][0] for m in monsters[:3]) / min(3, len(monsters))
                center_y = sum(m["center"][1] for m in monsters[:3]) / min(3, len(monsters))
                aoe_center = [center_x, center_y]
                
                actions.append({
                    "type": "use_skill",
                    "key": skill_key,
                    "target_position": aoe_center,
                    "target_count": min(3, len(monsters)),
                    "skill_type": "aoe",
                    "execution_priority": ActionPriority.NORMAL
                })
            else:
                # 单个怪物时直接瞄准
                actions.append({
                    "type": "use_skill",
                    "key": skill_key,
                    "target_position": target_center,
                    "target_id": target.get("id", "unknown"),
                    "target_type": target["class_name"],
                    "skill_type": "aoe",
                    "execution_priority": ActionPriority.NORMAL
                })
                
        else:  # 默认为近战技能
            # 近战技能 - 需要先接近目标
            actions.append({
                "type": "move_to",
                "position": target_center,
                "target_id": target.get("id", "unknown"),
                "target_type": target["class_name"],
                "purpose": "approach_target",
                "execution_priority": ActionPriority.HIGH
            })
            
            actions.append({
                "type": "use_skill",
                "key": skill_key,
                "target_position": target_center,
                "target_id": target.get("id", "unknown"),
                "target_type": target["class_name"],
                "skill_type": "melee",
                "execution_priority": ActionPriority.NORMAL
            })
    
    return actions

def generate_exploration_actions(detections, game_state):
    """
    生成探索状态下的操作 - 优化版
    
    参数:
        detections (list): 检测结果
        game_state (dict): 游戏状态
        
    返回:
        list: 操作指令
    """
    actions = []
    
    # 检查是否有门、NPC等交互物体
    doors = [d for d in detections if d["class_name"] == "door"]
    npcs = [d for d in detections if d["class_name"] == "npc"]
    items = [d for d in detections if d["class_name"] == "item"]
    
    # 优先拾取物品
    if items and not game_state.get("inventory_full", False):
        # 按照稀有度排序
        rarity_order = {
            "legendary": 0,
            "epic": 1,
            "rare": 2,
            "uncommon": 3,
            "common": 4,
            "unknown": 5
        }
        
        items.sort(key=lambda x: rarity_order.get(x.get("rarity", "unknown"), 999))
        
        # 选择最高稀有度的物品
        item = items[0]
        
        actions.append({
            "type": "move_to",
            "position": item["center"],
            "purpose": "pickup_item",
            "item_rarity": item.get("rarity", "unknown"),
            "execution_priority": ActionPriority.NORMAL
        })
        
        actions.append({
            "type": "interact",
            "key": "x",  # 假设X键是拾取键
            "purpose": "pickup_item",
            "execution_priority": ActionPriority.NORMAL
        })
    
    # 与NPC交互
    elif npcs and game_state.get("should_talk_to_npc", False):
        npc = npcs[0]
        actions.append({
            "type": "move_to",
            "position": npc["center"],
            "purpose": "talk_to_npc",
            "npc_id": npc.get("id", "unknown"),
            "execution_priority": ActionPriority.NORMAL
        })
        
        actions.append({
            "type": "interact",
            "key": "f",  # 假设F键是交互键
            "purpose": "talk_to_npc",
            "execution_priority": ActionPriority.NORMAL
        })
    
    # 前往下一个门
    elif doors:
        door = doors[0]
        actions.append({
            "type": "move_to",
            "position": door["center"],
            "purpose": "go_to_next_room",
            "door_id": door.get("id", "unknown"),
            "execution_priority": ActionPriority.NORMAL
        })
        
        actions.append({
            "type": "interact",
            "key": "f",  # 假设F键是交互键
            "purpose": "go_to_next_room",
            "execution_priority": ActionPriority.NORMAL
        })
    
    # 探索地图
    else:
        current_map = game_state.get("current_map", "")
        if current_map in DNF["maps"]:
            # 移动到地图出口
            exit_pos = DNF["maps"][current_map]["exit"]
            actions.append({
                "type": "move_to",
                "position": exit_pos,
                "purpose": "explore_map",
                "execution_priority": ActionPriority.NORMAL
            })
        else:
            # 智能探索 - 朝未探索区域移动
            explored_areas = game_state.get("explored_areas", [])
            
            if explored_areas:
                # 寻找未探索的方向
                directions = ["right", "up", "left", "down"]
                unexplored = [d for d in directions if d not in explored_areas]
                
                if unexplored:
                    direction = random.choice(unexplored)
                else:
                    direction = random.choice(directions)
            else:
                # 首次探索，默认向右
                direction = "right"
            
            actions.append({
                "type": "move_random",
                "direction": direction,
                "duration": random.uniform(1.0, 2.0),
                "purpose": "explore_unknown_map",
                "execution_priority": ActionPriority.NORMAL
            })
            
            # 记录已探索方向
            explored_areas = game_state.get("explored_areas", [])
            if direction not in explored_areas:
                explored_areas.append(direction)
                game_state["explored_areas"] = explored_areas
    
    return actions

def generate_pickup_actions(detections, game_state):
    """生成拾取物品的操作"""
    actions = []
    
    items = [d for d in detections if d["class_name"] == "item"]
    if items:
        # 按照稀有度排序
        rarity_order = {
            "legendary": 0,
            "epic": 1,
            "rare": 2,
            "uncommon": 3,
            "common": 4,
            "unknown": 5
        }
        
        items.sort(key=lambda x: rarity_order.get(x.get("rarity", "unknown"), 999))
        
        # 拾取最高稀有度的物品
        item = items[0]
        actions.append({
            "type": "move_to",
            "position": item["center"],
            "purpose": "pickup_item",
            "item_rarity": item.get("rarity", "unknown"),
            "execution_priority": ActionPriority.HIGH
        })
        
        # 连续点击以确保拾取
        for i in range(2):
            actions.append({
                "type": "interact",
                "key": "x",  # 假设X键是拾取键
                "purpose": "pickup_item",
                "execution_priority": ActionPriority.HIGH
            })
    
    return actions

def generate_talking_actions(detections, game_state):
    """生成对话状态下的操作"""
    actions = []
    
    dialog_boxes = [d for d in detections if d["class_name"] == "dialog"]
    if dialog_boxes:
        # 检查对话框中是否有选项
        dialog_options = [d for d in detections if d["class_name"] == "dialog_option"]
        
        if dialog_options:
            # 选择第一个选项（通常是继续任务）
            option = dialog_options[0]
            actions.append({
                "type": "click",
                "position": option["center"],
                "purpose": "select_dialog_option",
                "execution_priority": ActionPriority.NORMAL
            })
        else:
            # 点击空白区域继续对话
            actions.append({
                "type": "click",
                "position": [dialog_boxes[0]["center"][0], dialog_boxes[0]["center"][1] + 50],
                "purpose": "continue_dialog",
                "execution_priority": ActionPriority.NORMAL
            })
            
            # 或者按空格继续
            actions.append({
                "type": "interact",
                "key": "space",
                "purpose": "continue_dialog",
                "execution_priority": ActionPriority.NORMAL
            })
    
    return actions

def generate_lobby_actions(detections, game_state):
    """生成大厅状态下的操作"""
    actions = []
    
    # 检查是否有任务NPC或地下城入口
    npcs = [d for d in detections if d["class_name"] == "npc"]
    portals = [d for d in detections if d["class_name"] == "dungeon_portal"]
    
    if game_state.get("quest_active", False) and portals:
        # 有任务且发现地下城入口，进入地下城
        portal = portals[0]
        actions.append({
            "type": "move_to",
            "position": portal["center"],
            "purpose": "enter_dungeon",
            "execution_priority": ActionPriority.NORMAL
        })
        
        actions.append({
            "type": "interact",
            "key": "f",
            "purpose": "enter_dungeon",
            "execution_priority": ActionPriority.NORMAL
        })
    
    elif npcs:
        # 找任务NPC
        npc = npcs[0]
        actions.append({
            "type": "move_to",
            "position": npc["center"],
            "purpose": "talk_to_npc",
            "execution_priority": ActionPriority.NORMAL
        })
        
        actions.append({
            "type": "interact",
            "key": "f",
            "purpose": "talk_to_npc",
            "execution_priority": ActionPriority.NORMAL
        })
    
    else:
        # 随机移动
        actions.append({
            "type": "move_random",
            "direction": random.choice(["right", "left", "up", "down"]),
            "duration": random.uniform(0.5, 1.5),
            "purpose": "explore_lobby",
            "execution_priority": ActionPriority.LOW
        })
    
    return actions

def generate_dungeon_selection_actions(detections, game_state):
    """生成地下城选择界面的操作"""
    actions = []
    
    # 查找合适的地下城选项
    dungeon_options = [d for d in detections if d["class_name"] == "dungeon_option"]
    
    if dungeon_options:
        # 按照推荐级别排序
        recommended = [d for d in dungeon_options if d.get("is_recommended", False)]
        
        if recommended:
            option = recommended[0]
        else:
            option = dungeon_options[0]
        
        # 点击选择地下城
        actions.append({
            "type": "click",
            "position": option["center"],
            "purpose": "select_dungeon",
            "dungeon_name": option.get("text", "unknown"),
            "execution_priority": ActionPriority.NORMAL
        })
        
        # 点击确认按钮
        confirm_buttons = [d for d in detections if d["class_name"] == "confirm_button"]
        if confirm_buttons:
            actions.append({
                "type": "click",
                "position": confirm_buttons[0]["center"],
                "purpose": "confirm_dungeon",
                "execution_priority": ActionPriority.NORMAL
            })
    
    return actions

def generate_room_selection_actions(detections, game_state):
    """生成房间选择界面的操作"""
    actions = []
    
    # 查找房间选项
    room_options = [d for d in detections if d["class_name"] == "room_option"]
    
    if room_options:
        # 选择第一个房间
        option = room_options[0]
        
        actions.append({
            "type": "click",
            "position": option["center"],
            "purpose": "select_room",
            "execution_priority": ActionPriority.NORMAL
        })
        
        # 点击确认按钮
        confirm_buttons = [d for d in detections if d["class_name"] == "confirm_button"]
        if confirm_buttons:
            actions.append({
                "type": "click",
                "position": confirm_buttons[0]["center"],
                "purpose": "confirm_room",
                "execution_priority": ActionPriority.NORMAL
            })
    
    return actions

def generate_dead_actions(detections, game_state):
    """生成死亡状态下的操作"""
    actions = []
    
    # 查找复活按钮
    revive_buttons = [d for d in detections if d["class_name"] == "revive_button"]
    
    if revive_buttons:
        # 点击复活按钮
        actions.append({
            "type": "click",
            "position": revive_buttons[0]["center"],
            "purpose": "revive",
            "execution_priority": ActionPriority.CRITICAL
        })
    else:
        # 尝试快捷键复活
        actions.append({
            "type": "press_key_combo",
            "keys": ["alt", "r"],  # 假设Alt+R是复活快捷键
            "purpose": "revive",
            "execution_priority": ActionPriority.CRITICAL
        })
    
    return actions

def generate_default_actions(detections, game_state):
    """生成默认状态下的操作"""
    actions = []
    
    # 尝试按ESC退出可能的菜单
    actions.append({
        "type": "interact",
        "key": "escape",
        "purpose": "close_menu",
        "execution_priority": ActionPriority.HIGH
    })
    
    # 随机移动以探索
    actions.append({
        "type": "move_random",
        "direction": random.choice(["right", "left", "up", "down"]),
        "duration": random.uniform(0.5, 1.0),
        "purpose": "explore_unknown",
        "execution_priority": ActionPriority.LOW
    })
    
    return actions

def choose_skill(game_state, detections, monsters):
    """
    智能选择要使用的技能
    
    参数:
        game_state (dict): 游戏状态
        detections (list): 检测结果
        monsters (list): 怪物列表
        
    返回:
        tuple: (技能按键, 技能类型)
    """
    # 获取可用技能
    available_skills = list(DNF["skills"].keys())
    
    # 检查冷却中的技能
    cooldowns = game_state.get("cooldowns", {})
    available_skills = [s for s in available_skills if s not in cooldowns]
    
    if not available_skills:
        # 所有技能都在冷却中，使用普通攻击
        return "1", "melee"
    
    # 根据怪物情况选择最佳技能
    monster_count = len(monsters)
    
    # 技能类型
    skill_types = {
        "1": "melee",    # 普通攻击
        "2": "melee",    # 近战技能
        "3": "ranged",   # 远程技能
        "4": "aoe",      # AOE技能
        "5": "buff",     # 增益技能
        "6": "ultimate"  # 终极技能
    }
    
    # 对BOSS使用终极技能
    if any(m["class_name"] == "boss" for m in monsters) and "6" in available_skills:
        return "6", skill_types.get("6", "ultimate")
    
    # 多个怪物时使用AOE技能
    if monster_count >= 3 and "4" in available_skills:
        return "4", skill_types.get("4", "aoe")
    
    # 中等距离使用远程技能
    target_distance = estimate_distance(game_state, monsters[0])
    if target_distance > 150 and "3" in available_skills:
        return "3", skill_types.get("3", "ranged")
    
    # 近距离使用近战技能
    if target_distance < 100 and "2" in available_skills:
        return "2", skill_types.get("2", "melee")
    
    # 默认使用普通攻击
    return "1", skill_types.get("1", "melee")

def estimate_distance(game_state, target):
    """估算到目标的距离"""
    # 简单估算 - 实际情况可能需要更复杂的计算
    player_pos = [
        game_state.get("player_x", target["center"][0]),
        game_state.get("player_y", target["center"][1] - 100)  # 假设玩家在目标上方100像素
    ]
    
    dx = target["center"][0] - player_pos[0]
    dy = target["center"][1] - player_pos[1]
    
    return math.sqrt(dx*dx + dy*dy)

def humanize_actions(actions):
    """为动作添加人类行为特征 - 优化版"""
    previous_delay = 0
    
    for i, action in enumerate(actions):
        # 添加基础随机延迟
        base_delay = BEHAVIOR.get("min_delay", 0.1)
        max_delay = BEHAVIOR.get("max_delay", 0.8)
        
        # 不同动作类型有不同的延迟模式
        action_type = action.get("type", "")
        
        if action_type == "move_to":
            # 移动前的思考时间略长
            delay_mean = (base_delay + max_delay) / 2
            delay_std = (max_delay - base_delay) / 4
        elif action_type in ["use_skill", "interact"]:
            # 技能释放和交互通常更快
            delay_mean = base_delay * 1.5
            delay_std = base_delay / 2
        elif action_type == "click":
            # 点击通常很快
            delay_mean = base_delay
            delay_std = base_delay / 3
        else:
            # 默认延迟
            delay_mean = (base_delay + max_delay) / 2
            delay_std = (max_delay - base_delay) / 5
        
        # 生成延迟时间
        delay = random.normalvariate(delay_mean, delay_std)
        delay = max(base_delay, min(max_delay, delay))
        
        # 考虑动作连贯性 - 如果是连续相关的动作，延迟更短
        if i > 0 and are_actions_related(actions[i-1], action):
            delay *= 0.7
        
        # 优先级较高的动作延迟更短
        priority = action.get("execution_priority", ActionPriority.NORMAL)
        delay *= priority
        
        # 添加到动作中
        action["delay"] = delay + previous_delay
        previous_delay = 0  # 重置累积延迟
        
        # 为移动和点击添加随机偏移
        if action["type"] in ["move_to", "click", "use_skill"] and "position" in action:
            x, y = action["position"]
            
            # 计算适当的偏移量
            if "accuracy" in action:
                # 如果指定了精确度，使用它
                accuracy = action["accuracy"]
            elif action["type"] == "use_skill":
                # 技能瞄准通常更精确
                accuracy = 0.95
            else:
                # 默认精确度
                accuracy = 0.85
            
            max_offset_x = (1 - accuracy) * 20
            max_offset_y = (1 - accuracy) * 20
            
            x_offset = random.normalvariate(0, max_offset_x)
            y_offset = random.normalvariate(0, max_offset_y)
            
            action["position"] = [x + x_offset, y + y_offset]
        
        # 添加动作描述（用于日志和调试）
        if "purpose" in action and "description" not in action:
            action["description"] = f"{action['type']} - {action['purpose']}"

def are_actions_related(action1, action2):
    """判断两个动作是否相关联"""
    # 如果动作类型相同，可能是相关的
    if action1["type"] == action2["type"]:
        return True
    
    # 移动后接点击/技能/交互
    if action1["type"] == "move_to" and action2["type"] in ["click", "use_skill", "interact"]:
        return True
    
    # 技能后接移动（调整位置）
    if action1["type"] == "use_skill" and action2["type"] == "move_to":
        return True
    
    # 目的相同
    if "purpose" in action1 and "purpose" in action2 and action1["purpose"] == action2["purpose"]:
        return True
    
    return False

def record_decision(state, detections, actions):
    """记录决策历史"""
    decision = {
        "timestamp": time.time(),
        "state": state,
        "detection_count": len(detections),
        "action_count": len(actions),
        "action_types": [a["type"] for a in actions]
    }
    
    decision_history.append(decision)

def record_state_transition(from_state, to_state):
    """记录状态转换"""
    if from_state != to_state:
        transition_key = f"{from_state}->{to_state}"
        state_transitions[transition_key] = state_transitions.get(transition_key, 0) + 1