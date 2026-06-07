from __future__ import annotations

from llm_claw.models import AcquisitionTask, PlannedQuery


class DataNeedPlanner:
    def plan(self, task: AcquisitionTask) -> list[str]:
        return task.data_needed[:]


class QueryPlanner:
    def plan(self, task: AcquisitionTask, data_needs: list[str]) -> list[PlannedQuery]:
        entity = task.entity.display_name
        city = task.entity.city
        address = task.entity.address
        queries: list[PlannedQuery] = []
        for need in data_needs:
            terms = [entity, need]
            if city:
                terms.append(city)
            if address:
                terms.append(address)
            if task.source_policy.prefer_official_sources:
                terms.append("site:.gov OR city official")
            queries.append(PlannedQuery(text=" ".join(terms), data_need=need, freshness=task.freshness))

        base_terms = [part for part in [entity, city, address] if part]
        extras = ["planning commission agenda", "staff report PDF", "CEQA notice", "public comments"]
        for extra in extras:
            if len(queries) >= 10:
                break
            queries.append(PlannedQuery(text=" ".join(base_terms + [extra]), data_need=extra, freshness=task.freshness))
        return queries[:10]
