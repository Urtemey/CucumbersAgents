"""Router Agent - Маршрутизация жалоб по отделам."""

import time
from typing import Any, Dict, List

from complaintagents.base import BaseAgent, AgentResult
from complaintagents.models import (
    ComplaintMetrics,
    ComplaintCategory,
    UrgencyLevel,
    RoutingDecision,
)


class RouterAgent(BaseAgent):
    """
    Агент маршрутизации жалоб.
    
    Определяет:
    - Ответственный отдел/департамент
    - Роль ответственного сотрудника
    - Необходимость эскалации
    - SLA в зависимости от срочности
    """
    
    def __init__(self):
        super().__init__("RouterAgent")
        
        # Маппинг категорий на отделы
        self.department_map: Dict[ComplaintCategory, Dict] = {
            ComplaintCategory.MEDICAL: {
                "department": "medical_quality",
                "responsible_role": "quality_manager",
                "escalation_dept": "chief_medical_officer",
            },
            ComplaintCategory.SCHOOL: {
                "department": "education_quality",
                "responsible_role": "school_administrator",
                "escalation_dept": "education_board",
            },
            ComplaintCategory.HOUSING: {
                "department": "housing_services",
                "responsible_role": "housing_manager",
                "escalation_dept": "municipal_housing",
            },
            ComplaintCategory.SERVICE: {
                "department": "customer_service",
                "responsible_role": "service_manager",
                "escalation_dept": "operations_director",
            },
            ComplaintCategory.HOTEL: {
                "department": "guest_relations",
                "responsible_role": "front_office_manager",
                "escalation_dept": "hotel_director",
            },
            ComplaintCategory.RETAIL: {
                "department": "customer_experience",
                "responsible_role": "store_manager",
                "escalation_dept": "regional_manager",
            },
            ComplaintCategory.GOVERNMENT: {
                "department": "citizen_services",
                "responsible_role": "case_officer",
                "escalation_dept": "department_head",
            },
            ComplaintCategory.OTHER: {
                "department": "general_inquiries",
                "responsible_role": "case_manager",
                "escalation_dept": "supervisor",
            },
        }
        
        # Правила эскалации
        self.escalation_rules = [
            {
                "condition": lambda m: m.urgency == UrgencyLevel.CRITICAL,
                "reason": "Critical urgency requires immediate escalation",
            },
            {
                "condition": lambda m: m.toxicity_score > 0.8,
                "reason": "High toxicity - potential conflict situation",
            },
            {
                "condition": lambda m: m.severity_score > 0.9,
                "reason": "Severe issue requiring management attention",
            },
            {
                "condition": lambda m: m.contains_accusations and len(m.mentioned_persons) > 0,
                "reason": "Direct accusations against specific persons",
            },
            {
                "condition": lambda m: m.is_repeat_complaint,
                "reason": "Repeat complaint - previous resolution insufficient",
            },
        ]
    
    async def initialize(self) -> bool:
        """Инициализация агента."""
        self.log_start("Initializing router rules")
        return True
    
    async def process(self, metrics: ComplaintMetrics) -> AgentResult[RoutingDecision]:
        """
        Определить маршрутизацию для жалобы.
        """
        start_time = time.time()
        
        try:
            self.log_start(f"Routing for category: {metrics.category}")
            
            # Получаем базовый маппинг
            dept_config = self.department_map.get(
                metrics.category,
                self.department_map[ComplaintCategory.OTHER]
            )
            
            # Проверяем правила эскалации
            escalation_required = False
            escalation_reason = None
            
            for rule in self.escalation_rules:
                if rule["condition"](metrics):
                    escalation_required = True
                    escalation_reason = rule["reason"]
                    break
            
            # Определяем SLA
            sla_hours = metrics.urgency.sla_hours
            
            # Приоритетный буст
            priority_boost = (
                metrics.urgency in (UrgencyLevel.HIGH, UrgencyLevel.CRITICAL) or
                metrics.severity_score > 0.7
            )
            
            # Дополнительные отделы
            additional_departments = []
            
            if metrics.contains_pii:
                additional_departments.append("data_protection")
            
            if metrics.contains_accusations:
                additional_departments.append("hr_department")
                additional_departments.append("legal")
            
            if metrics.category == ComplaintCategory.MEDICAL and metrics.severity_score > 0.7:
                additional_departments.append("patient_safety")
            
            decision = RoutingDecision(
                department=dept_config["department"],
                responsible_role=dept_config["responsible_role"],
                escalation_required=escalation_required,
                escalation_reason=escalation_reason,
                sla_hours=sla_hours,
                priority_boost=priority_boost,
                additional_departments=additional_departments,
                routing_notes=self._generate_routing_notes(metrics),
            )
            
            processing_time = time.time() - start_time
            self.log_complete("routing", processing_time)
            
            result = AgentResult.ok(
                data=decision,
                processing_time=processing_time
            )
            
            if escalation_required:
                result.add_warning(f"Escalation required: {escalation_reason}")
            
            return result
            
        except Exception as e:
            self.log_error("routing", e)
            
            fallback = RoutingDecision(
                department="general_inquiries",
                responsible_role="case_manager",
                routing_notes="Fallback routing due to error",
            )
            return AgentResult.ok(data=fallback, processing_time=time.time() - start_time)
    
    def _generate_routing_notes(self, metrics: ComplaintMetrics) -> str:
        """Генерация заметок для маршрутизации."""
        notes = []
        
        if metrics.mentioned_persons:
            notes.append(f"Упомянуты: {', '.join(metrics.mentioned_persons)}")
        
        if metrics.mentioned_locations:
            notes.append(f"Локации: {', '.join(metrics.mentioned_locations)}")
        
        if metrics.mentioned_dates:
            notes.append(f"Даты: {', '.join(metrics.mentioned_dates)}")
        
        if metrics.keywords:
            notes.append(f"Ключевые слова: {', '.join(metrics.keywords[:5])}")
        
        return " | ".join(notes)
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверка состояния агента."""
        return {
            "name": self.name,
            "status": "ok",
            "departments_configured": len(self.department_map),
            "escalation_rules": len(self.escalation_rules),
            "initialized": self._initialized,
        }
    
    def add_department_mapping(self, category: ComplaintCategory, config: Dict):
        """Добавить/обновить маппинг категории на отдел."""
        self.department_map[category] = config
        self.logger.info(f"Updated department mapping for {category}")

