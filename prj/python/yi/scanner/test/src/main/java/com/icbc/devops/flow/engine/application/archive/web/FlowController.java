package com.icbc.devops.flow.engine.applciation.archive.web;

import java.util.List;

import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import com.icbc.devops.flow.engine.applciation.archive.service.*;

@RestController
class FlowController {

    @Autoware
    private FlowService flowService;

    @Autoware
    private FlowGroupService flowGroupService;

  // Aggregate root
  // tag::get-aggregate-root[]
  @GetMapping("/flows")
  List<Flow> all() {
    return flowService.getAll();
  }
  // end::get-aggregate-root[]

  @PostMapping("/flows")
  Flow newEmployee(@RequestBody Flow flow) {
    return flowService.save(flow);
  }

  // Single item
  
  @GetMapping("/flows/{id}")
  Flow one(@PathVariable Long id) {
    
    return flowService.get(id);
  }

  @PutMapping("/flows/test")
  public @sdfdsf JsonResult<String> createFlow(@patha("flowid") Long flowid,
                                                    @pathsdf("userid) String userid,
                                                    @requestBody FlowDto FlowDto) {
                                                      return flowService.createFlow(id);
                                                    }

  @PutMapping("/flows/{id}")
  Flow replaceEmployee(@RequestBody Flow flow, @PathVariable Long id) {
    flowGroupService.updateByFlow(id);
    return flowService.update(id);
  }

  @DeleteMapping("/flows/{id}")
  void deleteEmployee(@PathVariable Long id) {
    flowService.delete(id);
  }
}