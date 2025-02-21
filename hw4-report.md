# Cost-Benefit Analysis of Engaging Domes (Monthly Basis)

---

## 1. How much does it cost for Oz Sourcing to ship directly from Ovis to each city?

In **Scenario B**, Domes is *not* engaged (no local manufacturing). All demands are satisfied via **direct sea shipments** from Ovis to each Australian city. The model’s optimal solution yields:

- **Total Cost** = \$10,289,830

Under these constraints (no production in any Australian city), the solution confirms that **direct shipping from Ovis to each city** is the best possible approach. Specifically, the model sets:
- $ X_{ij} = 0 $ (no land shipments within Australia)
- $ z_i = 0 $ (no local production)
- All demand in each city is met by $ Y_j $ (units shipped from Ovis).

Hence, **10.29M** per month is the benchmark if Oz Sourcing relies solely on Ovis for production and ships directly to each city.

---

## 2. If Oz Sourcing does not engage Domes, is the current shipping practice best for Ovis?

Yes. When Domes is *not* engaged (i.e., no local manufacturing option), the model confirms that the **least-cost** solution is to ship directly from Ovis to each city. No further cost savings appear possible by routing through any single city or combining land shipments among the cities (the optimal solution indeed results in zero land shipments).

Thus, the **current direct-shipping practice** is validated as the best option if Domes is not involved.

---

## 3. In which city should Oz Sourcing ask Domes to manufacture?

In **Scenario A**, we allow Domes to manufacture in exactly one Australian city (i.e., $\sum_i z_i \le 1$). The model’s optimal solution chooses:

- **Brisbane** ($z_{\text{Brisbane}} = 1$)

Even though Adelaide has a slightly lower unit manufacturing cost (450 vs. 480 for Brisbane), Brisbane’s **geographic position** and **land-shipping costs** to other cities make it the overall cheapest production site. In other words, **Brisbane** strikes the best balance between production cost and distribution cost.

The resulting total monthly cost (with Brisbane producing 500 units for its own demand and some additional flows) is:

- **Total Cost** = \$7,426,700 

Within that solution, the model also decides to **ship a large portion of product via sea into Perth** (4,280 units), which is then redistributed by land to Adelaide, Melbourne, and Sydney. This combination leverages Perth’s favorable container-shipping cost from Ovis plus the local production in Brisbane.

---

## 4. How much could Oz Sourcing save, if engaging Domes proves to be cost-beneficial?

By comparing **Scenario A** (with Domes manufacturing in Brisbane) vs. **Scenario B** (no local manufacturing):

- **Cost with Domes (Scenario A):** 7,426,700  
- **Cost without Domes (Scenario B):** 10,289,830  

**Monthly Savings** = 10,289,830 − 7,426,700 ≈ **2,863,130**

Thus, **engaging Domes** in Brisbane reduces total monthly costs by nearly **2.86 million**. This significant cost reduction highlights the benefit of local production (in Brisbane) plus selective sea shipments (especially into Perth) versus direct shipment of all units from Ovis to each city.

---

