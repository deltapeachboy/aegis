"""
ポケモン情報関連のAPIエンドポイント
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional

from src.damage_calculator_api.schemas.responses import (
    PokemonInfo,
    MoveInfo,
    ItemInfo,
    TypeEffectivenessResponse,
    PokemonListResponse,
    MoveListResponse,
    ItemListResponse
)
from src.damage_calculator_api.utils.data_loader import get_data_loader
from src.damage_calculator_api.calculators.type_calculator import TypeCalculator

router = APIRouter()


@router.get("/list", response_model=PokemonListResponse)
async def get_pokemon_list(
    limit: Optional[int] = Query(None, description="取得件数上限"),
    offset: Optional[int] = Query(0, description="取得開始位置"),
    search: Optional[str] = Query(None, description="検索キーワード")
):
    """
    ポケモン一覧を取得
    
    検索機能付きでポケモン一覧を取得する
    """
    try:
        data_loader = get_data_loader()
        pokemon_names = list(data_loader.pokemon_data.keys())
        
        # 検索フィルタリング
        if search:
            pokemon_names = [name for name in pokemon_names if search in name]
        
        total = len(pokemon_names)
        
        # ページネーション
        if limit is not None:
            end_index = offset + limit
            pokemon_names = pokemon_names[offset:end_index]
        else:
            pokemon_names = pokemon_names[offset:]
        
        return PokemonListResponse(
            pokemon=pokemon_names,
            total=total
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ポケモン一覧取得中にエラーが発生しました: {str(e)}")


@router.get("/info/{pokemon_name}", response_model=PokemonInfo)
async def get_pokemon_info(pokemon_name: str):
    """
    指定されたポケモンの詳細情報を取得
    """
    try:
        data_loader = get_data_loader()
        
        if pokemon_name not in data_loader.pokemon_data:
            raise HTTPException(status_code=404, detail=f"ポケモン '{pokemon_name}' が見つかりません")
        
        pokemon_data = data_loader.pokemon_data[pokemon_name]
        
        # base_statsをdict形式に変換
        base_stats_dict = {
            "hp": pokemon_data.hp,
            "attack": pokemon_data.attack,
            "defense": pokemon_data.defense,
            "sp_attack": pokemon_data.sp_attack,
            "sp_defense": pokemon_data.sp_defense,
            "speed": pokemon_data.speed
        }
        
        return PokemonInfo(
            number=pokemon_data.number,
            name=pokemon_name,
            types=pokemon_data.types,
            abilities=pokemon_data.abilities,
            base_stats=base_stats_dict,
            weight=pokemon_data.weight
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ポケモン情報取得中にエラーが発生しました: {str(e)}")


@router.get("/moves", response_model=MoveListResponse)
async def get_move_list(
    limit: Optional[int] = Query(None, description="取得件数上限"),
    offset: Optional[int] = Query(0, description="取得開始位置"),
    search: Optional[str] = Query(None, description="検索キーワード"),
    move_type: Optional[str] = Query(None, description="技タイプでフィルタ")
):
    """
    技一覧を取得
    
    検索機能・タイプフィルタ付きで技一覧を取得する
    """
    try:
        data_loader = get_data_loader()
        move_names = list(data_loader.move_data.keys())
        
        # 検索フィルタリング
        if search:
            move_names = [name for name in move_names if search in name]
        
        # タイプフィルタリング
        if move_type:
            filtered_moves = []
            for name in move_names:
                move_data = data_loader.move_data[name]
                if move_data.move_type == move_type:
                    filtered_moves.append(name)
            move_names = filtered_moves
        
        total = len(move_names)
        
        # ページネーション
        if limit is not None:
            end_index = offset + limit
            move_names = move_names[offset:end_index]
        else:
            move_names = move_names[offset:]
        
        return MoveListResponse(
            moves=move_names,
            total=total
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"技一覧取得中にエラーが発生しました: {str(e)}")


@router.get("/moves/info/{move_name}", response_model=MoveInfo)
async def get_move_info(move_name: str):
    """
    指定された技の詳細情報を取得
    """
    try:
        data_loader = get_data_loader()
        
        if move_name not in data_loader.move_data:
            raise HTTPException(status_code=404, detail=f"技 '{move_name}' が見つかりません")
        
        move_data = data_loader.move_data[move_name]
        
        return MoveInfo(
            name=move_name,
            move_type=move_data.move_type,
            category=move_data.category,
            power=move_data.power,
            accuracy=move_data.accuracy,
            pp=move_data.pp
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"技情報取得中にエラーが発生しました: {str(e)}")


@router.get("/items", response_model=ItemListResponse)
async def get_item_list(
    limit: Optional[int] = Query(None, description="取得件数上限"),
    offset: Optional[int] = Query(0, description="取得開始位置"),
    search: Optional[str] = Query(None, description="検索キーワード")
):
    """
    道具一覧を取得
    
    検索機能付きで道具一覧を取得する
    """
    try:
        data_loader = get_data_loader()
        item_names = list(data_loader.item_data.keys())
        
        # 検索フィルタリング
        if search:
            item_names = [name for name in item_names if search in name]
        
        total = len(item_names)
        
        # ページネーション
        if limit is not None:
            end_index = offset + limit
            item_names = item_names[offset:end_index]
        else:
            item_names = item_names[offset:]
        
        return ItemListResponse(
            items=item_names,
            total=total
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"道具一覧取得中にエラーが発生しました: {str(e)}")


@router.get("/items/info/{item_name}", response_model=ItemInfo)
async def get_item_info(item_name: str):
    """
    指定された道具の詳細情報を取得
    """
    try:
        data_loader = get_data_loader()
        
        if item_name not in data_loader.item_data:
            raise HTTPException(status_code=404, detail=f"道具 '{item_name}' が見つかりません")
        
        item_data = data_loader.item_data[item_name]
        
        return ItemInfo(
            name=item_name,
            fling_power=item_data.fling_power,
            boost_type=item_data.boost_type,
            resist_type=item_data.resist_type,
            power_modifier=item_data.power_modifier,
            is_consumable=item_data.is_consumable
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"道具情報取得中にエラーが発生しました: {str(e)}")


@router.get("/type-effectiveness", response_model=TypeEffectivenessResponse)
async def get_type_effectiveness(
    attacking_type: str = Query(..., description="攻撃タイプ"),
    defending_type1: str = Query(..., description="防御タイプ1"),
    defending_type2: Optional[str] = Query(None, description="防御タイプ2（複合タイプの場合）")
):
    """
    タイプ相性を取得
    
    攻撃タイプと防御タイプ（複数）からタイプ相性を計算する
    """
    try:
        type_calculator = TypeCalculator()
        
        # 防御タイプのリストを作成
        defending_types = [defending_type1]
        if defending_type2:
            defending_types.append(defending_type2)
        
        # タイプ相性を計算
        effectiveness = type_calculator.calculate_type_effectiveness(
            attacking_type, defending_types
        )
        
        # 相性の説明を生成
        if effectiveness > 1.0:
            description = "効果抜群"
        elif effectiveness < 1.0:
            description = "効果いまひとつ"
        elif effectiveness == 0:
            description = "効果なし"
        else:
            description = "通常"
        
        return TypeEffectivenessResponse(
            attacking_type=attacking_type,
            defending_types=defending_types,
            effectiveness=effectiveness,
            description=description
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"タイプ相性計算中にエラーが発生しました: {str(e)}")