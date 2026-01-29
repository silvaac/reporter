"""Hyperliquid-specific portfolio monitoring implementation."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from hyperliquid.info import Info

from base.monitoring import PerformanceMetrics, PortfolioMonitor
from exceptions import ExchangeError

try:
    import pandas as pd
except ImportError:
    pd = None

logger = logging.getLogger(__name__)

def spot_tickers_here(coin="UETH", base='USDC',info=None):
    """
    Retrieves current tickers for a given coin.

    Args:
        coin (str, optional): Coin symbol (e.g. "ETH"). Defaults to "ETH".
        base (str, optional): Coin symbol (e.g. "USDC"). Defaults to "USDC".
        info (Info, optional): Hyperliquid Info client. If None, creates a new one.

    Returns:
        spot ticker (non intuitive symbol)
        Returns None if the API request fails or returns no data.

    Notes:
        - Requires Hyperliquid Info client to be initialized
    """
    if info is None:
        try:
            from token_data.hyperliquid import constants, setup
        except ImportError:
            return None

        _, info, _ = setup(base_url=constants.MAINNET_API_URL, skip_ws=True)

    maps = info.spot_meta_and_asset_ctxs()

    meta: dict[str, Any]
    if isinstance(maps, (list, tuple)) and maps:
        meta = maps[0]
    elif isinstance(maps, dict):
        meta = maps
    else:
        return None

    tokens = meta.get('tokens', [])
    universe = meta.get('universe', [])

    # Find the index of the coin and base in the universe of tokens
    # If not found, return None
    id_coin, id_base = None, None
    for token_ctx in tokens:
        if token_ctx.get('name') == coin:
            id_coin = token_ctx.get('index')
        elif token_ctx.get('name') == base:
            id_base = token_ctx.get('index')

    if id_coin is None or id_base is None:
        return None

    for ctx in universe:
        if ctx.get('tokens') == [id_coin, id_base]:
            return ctx.get('name')

    return None

class HyperliquidMonitor(PortfolioMonitor):
    """Portfolio monitoring for Hyperliquid exchange.
    
    Provides comprehensive portfolio tracking including:
    - Real-time performance metrics
    - Historical account value and P&L
    - Trade history and analysis
    - Funding rate tracking for perpetuals
    
    This class can work with token_data for enhanced functionality.
    
    Attributes:
        info: Hyperliquid Info API client.
        address: The account address to monitor.
    """
    
    def __init__(self, info: Info, address: str) -> None:
        """Initialize the Hyperliquid monitor.
        
        Args:
            info: Hyperliquid Info API client.
            address: The account address to monitor.
        """
        self._info = info
        self._address = address
        logger.info("Initialized HyperliquidMonitor for account %s", address[:10] + "...")
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current portfolio performance metrics.
        
        Returns:
            PerformanceMetrics with current values.
            
        Raises:
            ExchangeError: If unable to fetch metrics.
        """
        try:
            user_state = self._info.user_state(self._address)
            margin_summary = user_state.get('marginSummary', {})
            
            account_value = float(margin_summary.get('accountValue', 0))
            
            unrealized_pnl = 0.0
            if 'assetPositions' in user_state:
                for position_data in user_state['assetPositions']:
                    pos = position_data.get('position', {})
                    pnl = pos.get('unrealizedPnl')
                    if pnl:
                        unrealized_pnl += float(pnl)
            
            return PerformanceMetrics(
                account_value=account_value,
                unrealized_pnl=unrealized_pnl,
                timestamp=datetime.now(),
            )
        except Exception as e:
            raise ExchangeError(f"Failed to fetch current metrics: {e}") from e
    
    def get_portfolio_history(self, period: str = "allTime") -> dict[str, Any]:
        """Get historical portfolio data from Hyperliquid.
        
        Args:
            period: Time period for the data. Options:
                - "day": Last 24 hours
                - "week": Last 7 days
                - "month": Last 30 days
                - "allTime": All time (default)
                - "perpDay": Perpetuals only, last 24 hours
                - "perpWeek": Perpetuals only, last 7 days
                - "perpMonth": Perpetuals only, last 30 days
                - "perpAllTime": Perpetuals only, all time
        
        Returns:
            Dictionary containing:
            - accountValueHistory: List of [timestamp_ms, value_str] pairs
            - pnlHistory: List of [timestamp_ms, pnl_str] pairs
            - vlm: Total volume as string
        
        Raises:
            ExchangeError: If the portfolio data cannot be fetched.
            ValueError: If an invalid period is specified.
        """
        valid_periods = {
            "day", "week", "month", "allTime",
            "perpDay", "perpWeek", "perpMonth", "perpAllTime"
        }
        
        if period not in valid_periods:
            raise ValueError(
                f"Invalid period: {period}. Must be one of {valid_periods}"
            )
        
        try:
            portfolio_data = self._info.portfolio(self._address)
            
            for period_name, data in portfolio_data:
                if period_name == period:
                    return data
            
            logger.warning(f"Period '{period}' not found in portfolio data")
            return {
                "accountValueHistory": [],
                "pnlHistory": [],
                "vlm": "0.0"
            }
        except Exception as e:
            raise ExchangeError(f"Failed to fetch portfolio history: {e}") from e
    
    def get_portfolio_dataframe(
        self,
        period: str = "allTime",
        data_type: str = "both",
    ) -> Any:
        """Get portfolio history as a pandas DataFrame.
        
        Args:
            period: Time period for the data.
            data_type: Type of data to include ("both", "account_value", "pnl").
        
        Returns:
            pandas.DataFrame with datetime index and columns.
            
        Raises:
            ImportError: If pandas is not installed.
            ValueError: If an invalid period or data_type is specified.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for get_portfolio_dataframe(). "
                "Install it with: pip install pandas"
            ) from e
        
        valid_data_types = {"both", "account_value", "pnl"}
        if data_type not in valid_data_types:
            raise ValueError(
                f"Invalid data_type: {data_type}. Must be one of {valid_data_types}"
            )
        
        history = self.get_portfolio_history(period)
        
        account_value_data = []
        if data_type in {"both", "account_value"}:
            for timestamp_ms, value_str in history.get("accountValueHistory", []):
                account_value_data.append({
                    "timestamp": pd.to_datetime(timestamp_ms, unit='ms'),
                    "account_value": float(value_str)
                })
        
        pnl_data = []
        if data_type in {"both", "pnl"}:
            for timestamp_ms, pnl_str in history.get("pnlHistory", []):
                pnl_data.append({
                    "timestamp": pd.to_datetime(timestamp_ms, unit='ms'),
                    "pnl": float(pnl_str)
                })
        
        if data_type == "both":
            if account_value_data and pnl_data:
                df_account = pd.DataFrame(account_value_data).set_index("timestamp")
                df_pnl = pd.DataFrame(pnl_data).set_index("timestamp")
                df = df_account.join(df_pnl, how="outer")
            elif account_value_data:
                df = pd.DataFrame(account_value_data).set_index("timestamp")
                df["pnl"] = 0.0
            elif pnl_data:
                df = pd.DataFrame(pnl_data).set_index("timestamp")
                df["account_value"] = 0.0
            else:
                df = pd.DataFrame(columns=["account_value", "pnl"])
                df.index.name = "timestamp"
        elif data_type == "account_value":
            if account_value_data:
                df = pd.DataFrame(account_value_data).set_index("timestamp")
            else:
                df = pd.DataFrame(columns=["account_value"])
                df.index.name = "timestamp"
        else:
            if pnl_data:
                df = pd.DataFrame(pnl_data).set_index("timestamp")
            else:
                df = pd.DataFrame(columns=["pnl"])
                df.index.name = "timestamp"
        
        df = df.sort_index()
        return df
    
    def get_portfolio_summary(self) -> dict[str, Any]:
        """Get a comprehensive summary of portfolio performance.
        
        Returns:
            Dictionary with keys for each time period containing:
            - latest_account_value: Most recent account value (float)
            - total_pnl: Total P&L for the period (float)
            - volume: Trading volume for the period (float)
            - data_points: Number of historical data points available (int)
        
        Raises:
            ExchangeError: If the portfolio data cannot be fetched.
        """
        try:
            portfolio_data = self._info.portfolio(self._address)
            
            summary = {}
            for period_name, data in portfolio_data:
                account_history = data.get("accountValueHistory", [])
                pnl_history = data.get("pnlHistory", [])
                volume = float(data.get("vlm", "0.0"))
                
                latest_account_value = 0.0
                if account_history:
                    latest_account_value = float(account_history[-1][1])
                
                total_pnl = 0.0
                if pnl_history:
                    total_pnl = float(pnl_history[-1][1])
                
                summary[period_name] = {
                    "latest_account_value": latest_account_value,
                    "total_pnl": total_pnl,
                    "volume": volume,
                    "data_points": len(account_history)
                }
            
            return summary
        except Exception as e:
            raise ExchangeError(f"Failed to fetch portfolio summary: {e}") from e
    
    def get_trade_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        coin: Optional[str] = None,
        aggregated: bool = False,
        as_dataframe: bool = False,
    ) -> Any:
        """Get historical trade data.
        
        Uses token_data's get_user_fills for reliable data retrieval.
        
        Args:
            start_time: Optional start time for filtering trades.
            end_time: Optional end time for filtering trades.
            coin: Optional coin symbol to filter trades.
            aggregated: If True, returns aggregated fills.
            as_dataframe: If True, returns pandas DataFrame instead of list.
        
        Returns:
            List of trade dictionaries or pandas DataFrame with columns:
            - coin: Asset symbol
            - px: Fill price
            - sz: Fill size
            - side: Trade side (A for ask/sell, B for bid/buy)
            - time: Original timestamp in milliseconds
            - startPosition: Position size before fill
            - dir: Direction of trade
            - closedPnl: Realized PnL from closing position
            - fee: Trading fee paid
            - datetime: Timestamp (DataFrame index)
        
        Raises:
            ExchangeError: If unable to fetch trade history.
        """
        try:
            from token_data.hyper_account import get_user_fills
            
            # Get fills as DataFrame
            fills_df = get_user_fills(
                self._address,
                info=self._info,
                aggregated=aggregated
            )
            
            if fills_df is None or fills_df.empty:
                return pd.DataFrame() if as_dataframe else []
            
            # Apply filters
            if start_time is not None:
                fills_df = fills_df[fills_df.index >= start_time]
            if end_time is not None:
                fills_df = fills_df[fills_df.index <= end_time]
            if coin is not None:
                fills_df = fills_df[fills_df['coin'] == coin]
            
            if as_dataframe:
                return fills_df
            
            # Convert to list of dictionaries
            trades = []
            for idx, row in fills_df.iterrows():
                trade = row.to_dict()
                trade['datetime'] = idx
                trades.append(trade)
            
            return trades
        except ImportError as e:
            raise ExchangeError(
                "token_data package required for trade history. "
                "Install with: pip install token_data"
            ) from e
        except Exception as e:
            raise ExchangeError(f"Failed to fetch trade history: {e}") from e
    
    def get_funding_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        coin: Optional[str] = None,
        lookback: Optional[int] = None,
        as_dataframe: bool = False,
    ) -> Any:
        """Get funding rate history for perpetual positions.
        
        Uses token_data's get_user_funding_history for reliable data retrieval.
        
        Args:
            start_time: Optional start time for filtering (datetime object).
            end_time: Optional end time for filtering (datetime object).
            coin: Optional coin symbol to filter.
            lookback: Number of days to look back from end_time if start_time is None.
                     Defaults to None (gets all available history).
            as_dataframe: If True, returns pandas DataFrame instead of list.
        
        Returns:
            List of funding payment dictionaries or pandas DataFrame with columns:
            - coin: Asset symbol
            - usdc: USD value of funding payment (positive = received, negative = paid)
            - szi: Position size
            - fundingRate: The funding rate
            - datetime: Timestamp (DataFrame index)
            - time: Timestamp in milliseconds (list only)
        
        Raises:
            ExchangeError: If unable to fetch funding history.
        """
        try:
            from token_data.hyper_account import get_user_funding_history
            
            # Convert datetime objects to ISO strings for token_data API
            start_date = start_time.isoformat() if start_time else None
            end_date = end_time.isoformat() if end_time else None
            
            # Get funding history as DataFrame
            funding_df = get_user_funding_history(
                self._address,
                start_date=start_date,
                end_date=end_date,
                lookback=lookback if lookback is not None else 3650,  # 10 years for all-time data
                info=self._info
            )
            
            if funding_df is None or funding_df.empty:
                return pd.DataFrame() if as_dataframe else []
            
            # Apply additional time filters if specified (client-side filtering)
            # This handles cases where the API returns more data than requested
            if start_time is not None:
                funding_df = funding_df[funding_df.index >= start_time]
            if end_time is not None:
                funding_df = funding_df[funding_df.index <= end_time]
            
            # Apply coin filter if specified
            if coin is not None:
                funding_df = funding_df[funding_df['coin'] == coin]
            
            if as_dataframe:
                return funding_df
            
            # Convert DataFrame to list of dictionaries
            payments = []
            for idx, row in funding_df.iterrows():
                payment = {
                    'coin': row['coin'],
                    'usdc': float(row['usdc']),
                    'szi': float(row['szi']),
                    'fundingRate': float(row['fundingRate']),
                    'datetime': idx,
                    'time': int(idx.timestamp() * 1000)
                }
                payments.append(payment)
            
            return payments
        except ImportError as e:
            raise ExchangeError(
                "token_data package required for funding history. "
                "Install with: pip install token_data"
            ) from e
        except Exception as e:
            raise ExchangeError(f"Failed to fetch funding history: {e}") from e
    
    def get_account_summary(
        self,
        lookback_days: int = 3650,
    ) -> dict[str, Any]:
        """Get comprehensive account summary including deposits, withdrawals, and current value.
        
        Uses token_data's get_account_summary for reliable data retrieval.
        
        Args:
            lookback_days: Number of days to look back for ledger history.
                          Defaults to 3650 (~10 years).
        
        Returns:
            Dictionary containing:
            - total_deposits: Total amount deposited in USDC
            - total_withdrawals: Total amount withdrawn in USDC
            - net_deposits: Net deposits (deposits - withdrawals) in USDC
            - current_value: Current total account value in USDC
            - spot_value: Value in spot account in USDC
            - perp_value: Value in perpetual account in USDC
            - perp_position_value: Current notional value of perpetual positions in USDC
            - total_pnl: Total profit/loss (current_value - net_deposits)
            - pnl_percentage: P&L as percentage of net deposits
            - unrealized_pnl: Unrealized P&L from open perpetual positions
            - cash_in_perp: Cash in perpetual account
            - when: Timestamp when summary was generated
        
        Raises:
            ExchangeError: If unable to fetch account summary.
        """
        def _safe_float(val: Any, default: float = 0.0) -> float:
            try:
                return float(val)
            except Exception:
                return default

        def _fallback_summary() -> dict[str, Any]:
            user_state = self._info.user_state(self._address)
            margin_summary = user_state.get("marginSummary", {})

            perp_value = _safe_float(margin_summary.get("accountValue", 0.0) or 0.0)

            perp_position_value = 0.0
            unrealized_pnl = 0.0
            for position_data in user_state.get("assetPositions", []) or []:
                pos = position_data.get("position", {}) or {}
                try:
                    perp_position_value += _safe_float(pos.get("positionValue", pos.get("value", 0.0)) or 0.0)
                except Exception:
                    pass
                try:
                    unrealized_pnl += _safe_float(pos.get("unrealizedPnl", 0.0) or 0.0)
                except Exception:
                    pass

            # Get total account value from portfolio history
            # accountValueHistory includes both perp and spot values
            total_account_value = 0.0
            try:
                portfolio_data = self._info.portfolio(self._address)
                for period_name, data in portfolio_data:
                    if period_name == "allTime":
                        account_history = data.get("accountValueHistory", [])
                        if account_history:
                            # Get the most recent value
                            latest = account_history[-1]
                            total_account_value = _safe_float(latest[1])
                            break
            except Exception as e:
                logger.warning(f"Failed to get account value history: {e}")
            
            # Calculate spot value as: total - perp
            # This is more accurate than trying to value individual spot tokens
            spot_value = max(0.0, total_account_value - perp_value)
            current_value = float(total_account_value if total_account_value > 0 else perp_value)
            
            # Calculate spot unrealized P&L
            # For each non-USDC token: unrealized = current_value - entry_cost
            # USDC has no unrealized P&L (it's the base currency)
            spot_unrealized_pnl = 0.0
            try:
                spot_state = self._info.spot_user_state(self._address)
                usdc_balance = 0.0
                
                for balance in spot_state.get('balances', []):
                    coin = balance.get('coin', '')
                    entry_ntl = _safe_float(balance.get('entryNtl', 0.0))
                    
                    if coin == 'USDC':
                        # USDC is the base currency, track its balance
                        usdc_balance = _safe_float(balance.get('total', 0.0))
                    else:
                        # For non-USDC tokens: current value = (spot_value - USDC)
                        # But we need to calculate per-token, so use entry cost as proxy
                        # Since we can't easily get individual token current values,
                        # we calculate: (total_spot - USDC) - sum(non-USDC entry costs)
                        pass
                
                # Calculate total entry cost for non-USDC tokens
                total_non_usdc_entry = sum(
                    _safe_float(b.get('entryNtl', 0.0)) 
                    for b in spot_state.get('balances', []) 
                    if b.get('coin') != 'USDC'
                )
                
                # Current value of non-USDC tokens = total spot value - USDC balance
                non_usdc_current_value = spot_value - usdc_balance
                
                # Unrealized P&L on non-USDC tokens
                spot_unrealized_pnl = non_usdc_current_value - total_non_usdc_entry
                
                logger.info(f"Spot: total=${spot_value:.2f}, USDC=${usdc_balance:.2f}, non-USDC current=${non_usdc_current_value:.2f}, entry=${total_non_usdc_entry:.2f}, unrealized=${spot_unrealized_pnl:.2f}")
            except Exception as e:
                logger.warning(f"Failed to calculate spot unrealized P&L: {e}")
            
            # Total unrealized P&L = perp unrealized + spot unrealized
            unrealized_pnl += spot_unrealized_pnl
            
            logger.info(f"Total account value: ${total_account_value:.2f}, Perp: ${perp_value:.2f}, Spot: ${spot_value:.2f}, Unrealized P&L: ${unrealized_pnl:.2f}")

            # Fetch deposits and withdrawals from ledger
            total_deposits = 0.0
            total_withdrawals = 0.0
            
            try:
                # Calculate start time based on lookback_days
                from datetime import timedelta
                end_time_ms = int(datetime.now().timestamp() * 1000)
                start_time_ms = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
                
                # Fetch non-funding ledger updates (includes deposits, withdrawals, transfers)
                ledger_updates = self._info.user_non_funding_ledger_updates(
                    self._address,
                    start_time_ms,
                    end_time_ms
                )
                
                for update in ledger_updates:
                    delta = update.get("delta", {})
                    
                    # Handle deposit
                    if "type" in delta and delta["type"] == "deposit":
                        usdc_amount = _safe_float(delta.get("usdc", 0.0))
                        total_deposits += usdc_amount
                    
                    # Handle withdrawal
                    elif "type" in delta and delta["type"] == "withdraw":
                        usdc_amount = _safe_float(delta.get("usdc", 0.0))
                        total_withdrawals += abs(usdc_amount)
                    
                    # Handle internal transfers (subAccountTransfer)
                    elif "type" in delta and delta["type"] == "subAccountTransfer":
                        usdc_amount = _safe_float(delta.get("usdc", 0.0))
                        if usdc_amount > 0:
                            total_deposits += usdc_amount
                        else:
                            total_withdrawals += abs(usdc_amount)
                
                logger.info(f"Fetched ledger data: deposits=${total_deposits:.2f}, withdrawals=${total_withdrawals:.2f}")
            
            except Exception as e:
                logger.warning(f"Failed to fetch ledger updates in fallback: {e}")
                # Continue with 0.0 values if ledger fetch fails

            net_deposits = total_deposits - total_withdrawals
            total_pnl = current_value - net_deposits
            pnl_percentage = (total_pnl / net_deposits * 100) if net_deposits > 0 else 0.0

            summary: dict[str, Any] = {
                "total_deposits": total_deposits,
                "total_withdrawals": total_withdrawals,
                "net_deposits": net_deposits,
                "current_value": current_value,
                "spot_value": spot_value,
                "perp_value": perp_value,
                "perp_position_value": perp_position_value,
                "total_pnl": total_pnl,
                "pnl_percentage": pnl_percentage,
                "unrealized_pnl": unrealized_pnl,
                "cash_in_perp": _safe_float(margin_summary.get("availableBalance", 0.0) or 0.0),
                "when": None,
            }

            try:
                summary["when"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                summary["when"] = "N/A"

            return summary

        try:
            from token_data.hyper_account import get_account_summary as td_get_account_summary

            result = td_get_account_summary(
                self._address,
                info=self._info,
                lookback_days=lookback_days,
            )

            if isinstance(result, tuple):
                # Expected historical behavior: (summary_dict, details)
                if len(result) >= 1 and isinstance(result[0], dict):
                    return result[0]
                raise TypeError("Unexpected tuple return from token_data.get_account_summary")

            if isinstance(result, dict):
                # Some versions may return only the summary dict
                return result

            # Some buggy versions return None after printing errors.
            raise TypeError("token_data.get_account_summary returned None")

        except ImportError:
            return _fallback_summary()
        except NameError:
            # Some token_data versions reference helper functions that may not exist
            # depending on optional data feeds (e.g. spot tickers). Do not fail hard.
            return _fallback_summary()
        except Exception:
            logger.exception("token_data account summary failed; falling back to Info-based summary")
            try:
                return _fallback_summary()
            except Exception as e:
                raise ExchangeError(f"Failed to fetch account summary: {e}") from e
    
    def analyze_performance(self, period: str = "allTime") -> dict[str, float]:
        """Analyze portfolio performance metrics.
        
        Args:
            period: Time period to analyze.
        
        Returns:
            Dictionary with calculated performance metrics:
            - total_return: Total return percentage
            - total_pnl: Total profit/loss
            - volume: Trading volume
            - num_trades: Number of trades
            - avg_trade_size: Average trade size
        """
        try:
            history = self.get_portfolio_history(period)
            
            account_history = history.get("accountValueHistory", [])
            pnl_history = history.get("pnlHistory", [])
            volume = float(history.get("vlm", "0.0"))
            
            total_pnl = 0.0
            if pnl_history:
                total_pnl = float(pnl_history[-1][1])
            
            total_return = 0.0
            if account_history and len(account_history) > 1:
                initial_value = float(account_history[0][1])
                final_value = float(account_history[-1][1])
                if initial_value > 0:
                    total_return = ((final_value - initial_value) / initial_value) * 100
            
            trades = self.get_trade_history()
            num_trades = len(trades)
            avg_trade_size = volume / num_trades if num_trades > 0 else 0.0
            
            return {
                "total_return": total_return,
                "total_pnl": total_pnl,
                "volume": volume,
                "num_trades": num_trades,
                "avg_trade_size": avg_trade_size,
            }
        except Exception as e:
            logger.exception("Failed to analyze performance")
            return {
                "total_return": 0.0,
                "total_pnl": 0.0,
                "volume": 0.0,
                "num_trades": 0,
                "avg_trade_size": 0.0,
            }
