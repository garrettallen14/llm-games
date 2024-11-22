from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import re
import numpy as np
from numpy import npv, irr

@dataclass
class CellValue:
    """Represents a cell's value and metadata"""
    value: Union[float, str] = ""
    formula: str = ""
    format: str = "general"  # general, currency, percentage
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class UnderwritingModel:
    """DCF model for property underwriting"""
    
    def __init__(self):
        self.cells: Dict[str, CellValue] = {}
        self.history: List[str] = []
        
        # Model assumptions
        self.assumptions = {
            "revenue_growth": 0.03,    # 3% annual growth
            "opex_growth": 0.02,       # 2% annual growth
            "start_occupancy": 0.85,   # 85% initial
            "target_occupancy": 0.92,  # 92% stabilized
            "capex_psf": 2.0,         # $2/sf annual
            "discount_rate": 0.08,     # 8% discount rate
            "exit_cap": 0.065,        # 6.5% exit cap
            "square_feet": 50000,     # 50,000 SF
        }
        
        # Initialize template
        self._create_template()
    
    def _create_template(self):
        """Initialize the model template"""
        # Headers
        headers = {
            'A0': 'Year',
            'B0': 'Revenue',
            'C0': 'Occupancy',
            'D0': 'Effective Revenue',
            'E0': 'OpEx',
            'F0': 'NOI',
            'G0': 'CapEx',
            'H0': 'Free Cash Flow',
            'I0': 'Terminal Value',
            'J0': 'Total Cash Flow',
            'K0': 'Discount Factor',
            'L0': 'PV of Cash Flow',
            'M0': 'Property Value'
        }
        
        for cell, value in headers.items():
            self.cells[cell] = CellValue(value=value)
        
        # Years
        for i in range(1, 6):
            self.cells[f'A{i}'] = CellValue(value=i)
    
    def process_command(self, command: str) -> Tuple[bool, str]:
        """Process a single command"""
        self.history.append(command)
        
        # INSERT: value cell
        insert_match = re.match(r'^INSERT:\s*(-?\d*\.?\d*)\s+([A-M][0-9]{1,2})$', command)
        if insert_match:
            value, cell = insert_match.groups()
            return self.insert_value(float(value), cell)
        
        # FORMULA: cell expression
        formula_match = re.match(r'^FORMULA:\s*([A-M][0-9]{1,2})\s+(.+)$', command)
        if formula_match:
            cell, expr = formula_match.groups()
            return self.set_formula(cell, expr)
        
        # NPV: range rate_cell
        npv_match = re.match(r'^NPV:\s*([A-M][0-9]{1,2}):([A-M][0-9]{1,2})\s+([A-M][0-9]{1,2})$', command)
        if npv_match:
            start, end, rate_cell = npv_match.groups()
            return self.calculate_npv(start, end, rate_cell)
        
        return False, "Invalid command format"
    
    def insert_value(self, value: float, cell: str) -> Tuple[bool, str]:
        """Insert a value into a cell"""
        if not self._is_valid_cell(cell):
            return False, f"Invalid cell reference: {cell}"
        
        self.cells[cell] = CellValue(value=value)
        return True, f"Inserted {value} into {cell}"
    
    def set_formula(self, cell: str, expr: str) -> Tuple[bool, str]:
        """Set a formula for a cell"""
        if not self._is_valid_cell(cell):
            return False, f"Invalid cell reference: {cell}"
        
        try:
            # Basic formula evaluation - would need more sophisticated parsing
            # for a real implementation
            result = eval(expr)
            self.cells[cell] = CellValue(value=result, formula=expr)
            return True, f"Set formula {expr} in {cell}, result: {result}"
        except:
            return False, f"Invalid formula: {expr}"
    
    def calculate_npv(self, start: str, end: str, rate_cell: str) -> Tuple[bool, str]:
        """Calculate NPV of a range"""
        if not all(self._is_valid_cell(c) for c in [start, end, rate_cell]):
            return False, "Invalid cell reference"
        
        try:
            rate = float(self.cells[rate_cell].value)
            values = self._get_range_values(start, end)
            result = npv(rate, values)
            return True, f"NPV: {result}"
        except:
            return False, "Error calculating NPV"
    
    def _is_valid_cell(self, cell: str) -> bool:
        """Check if cell reference is valid"""
        return bool(re.match(r'^[A-M][0-9]{1,2}$', cell))
    
    def _get_range_values(self, start: str, end: str) -> List[float]:
        """Get values from a cell range"""
        start_col = ord(start[0]) - ord('A')
        start_row = int(start[1:])
        end_col = ord(end[0]) - ord('A')
        end_row = int(end[1:])
        
        values = []
        for row in range(start_row, end_row + 1):
            for col in range(start_col, end_col + 1):
                cell = f"{chr(ord('A') + col)}{row}"
                if cell in self.cells:
                    values.append(float(self.cells[cell].value))
        
        return values
    
    def is_complete(self) -> Tuple[bool, str]:
        """Check if model is complete and correct"""
        checks = [
            self._check_revenue_growth(),
            self._check_occupancy(),
            self._check_calculations(),
            self._check_final_value()
        ]
        
        success = all(check[0] for check in checks)
        messages = "\n".join(check[1] for check in checks)
        return success, messages
    
    def _check_revenue_growth(self) -> Tuple[bool, str]:
        """Verify revenue growth assumptions"""
        try:
            # Check years 1-5
            for i in range(1, 5):
                current = float(self.cells[f'B{i}'].value)
                next_val = float(self.cells[f'B{i+1}'].value)
                growth = (next_val - current) / current
                if abs(growth - self.assumptions['revenue_growth']) > 0.001:
                    return False, f"Revenue growth in year {i} is {growth:.1%}, should be {self.assumptions['revenue_growth']:.1%}"
            return True, "Revenue growth verified"
        except:
            return False, "Could not verify revenue growth"
    
    def _check_occupancy(self) -> Tuple[bool, str]:
        """Verify occupancy stabilization"""
        try:
            # Check initial occupancy
            start_occ = float(self.cells['C1'].value)
            if abs(start_occ - self.assumptions['start_occupancy']) > 0.001:
                return False, f"Initial occupancy should be {self.assumptions['start_occupancy']:.1%}"
            
            # Check stabilized occupancy by year 3
            for i in range(3, 6):
                occ = float(self.cells[f'C{i}'].value)
                if abs(occ - self.assumptions['target_occupancy']) > 0.001:
                    return False, f"Year {i} occupancy should be {self.assumptions['target_occupancy']:.1%}"
            
            return True, "Occupancy stabilization verified"
        except:
            return False, "Could not verify occupancy"
    
    def _check_calculations(self) -> Tuple[bool, str]:
        """Verify calculation accuracy"""
        try:
            # Check each year
            for i in range(1, 6):
                # Effective Revenue = Revenue * Occupancy
                rev = float(self.cells[f'B{i}'].value)
                occ = float(self.cells[f'C{i}'].value)
                eff_rev = float(self.cells[f'D{i}'].value)
                if abs(rev * occ - eff_rev) > 0.01:
                    return False, f"Year {i} effective revenue calculation incorrect"
                
                # NOI = Effective Revenue - OpEx
                opex = float(self.cells[f'E{i}'].value)
                noi = float(self.cells[f'F{i}'].value)
                if abs(eff_rev - opex - noi) > 0.01:
                    return False, f"Year {i} NOI calculation incorrect"
                
                # CapEx = $2/SF
                capex = float(self.cells[f'G{i}'].value)
                if abs(capex - self.assumptions['capex_psf'] * self.assumptions['square_feet']) > 0.01:
                    return False, f"Year {i} CapEx calculation incorrect"
                
                # Free Cash Flow = NOI - CapEx
                fcf = float(self.cells[f'H{i}'].value)
                if abs(noi - capex - fcf) > 0.01:
                    return False, f"Year {i} free cash flow calculation incorrect"
            
            # Check terminal value
            year5_noi = float(self.cells['F5'].value)
            term_val = float(self.cells['I5'].value)
            expected_tv = year5_noi / self.assumptions['exit_cap']
            if abs(term_val - expected_tv) > 0.01:
                return False, "Terminal value calculation incorrect"
            
            return True, "All calculations verified"
        except:
            return False, "Could not verify calculations"
    
    def _check_final_value(self) -> Tuple[bool, str]:
        """Verify final property value calculation"""
        try:
            # Get cash flows
            cash_flows = []
            for i in range(1, 6):
                fcf = float(self.cells[f'H{i}'].value)
                if i == 5:
                    term_val = float(self.cells['I5'].value)
                    fcf += term_val
                cash_flows.append(fcf)
            
            # Calculate NPV
            value = npv(self.assumptions['discount_rate'], cash_flows)
            
            # Check final value in M0
            final_value = float(self.cells['M0'].value)
            if abs(value - final_value) > 0.01:
                return False, f"Final value incorrect. Should be {value:,.2f}"
            
            return True, "Final value verified"
        except:
            return False, "Could not verify final value"
